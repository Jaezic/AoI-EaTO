# workspace/optimization/hovering_pos_optimization.py
import numpy as np
import cvxpy as cp
from typing import Tuple, Optional, List

from model.uav import UAV
from model.environment import Environment
from model.trajectory import TrajectorySolution
from configs.data_classes import SimulationConfig
from optimization.hovering_pos_constraints import HoveringPositionConstraintManager
from utils import data_logger as logger

class HoveringPositionOptimizerSCA:
    """
    Algorithm 3: SCA-Based Hovering Position Optimization for P2.2.
    Given a fixed visiting sequence and flight speed, find the optimal hovering positions (q̃_k, p_k) and
    related slack variables to minimize the total mission completion time.
    """
    uav: UAV
    env: Environment
    sim_cfg: SimulationConfig
    
    # In P2.2, the visiting sequence and speed are fixed
    fixed_trajectory_solution_for_hover_opt: TrajectorySolution

    max_sca_iterations: int
    sca_convergence_tol: float
    
    num_areas: int

    def __init__(self,
                 uav_model: UAV,
                 env_model: Environment,
                 sim_config: SimulationConfig,
                 fixed_trajectory_params: TrajectorySolution, # Using visiting sequence and speed information
                 max_sca_iterations: int = 10,
                 sca_convergence_tol: float = 1e-3):
        self.uav = uav_model
        self.env = env_model
        self.sim_cfg = sim_config
        self.fixed_trajectory_solution_for_hover_opt = fixed_trajectory_params
        self.max_sca_iterations = max_sca_iterations
        self.sca_convergence_tol = sca_convergence_tol
        self.num_areas = env_model.get_total_monitoring_areas()

    def _calculate_total_time_objective_p221_48a(self,
                                                 mu_k_vars: cp.Variable,    # (N,)
                                                 beta_k_vars: cp.Variable,   # (N,)
                                                 eta_k_vars: cp.Variable,    # (N,)
                                                 psi_k_vars: cp.Variable,    # (N,)
                                                 # p_k_vars는 t_f^0 계산 시 필요 (마지막 p_k)
                                                 # 여기서는 p_k_vars 대신 current_p_all_l 사용
                                                 current_p_all_l_for_tf0: np.ndarray
                                                ) -> cp.Expression:
        """Returns the objective function (equation 48a) of P2.2.1 as a CVXPY expression."""
        total_time_expr = 0.0
        
        v0_fixed = self.fixed_trajectory_solution_for_hover_opt.speeds_v_mps[0]
        
        S_k_bits = self.uav.config.communication.data_packet_size_Sk_mbits * 1e6
        B_hz = self.uav.config.communication.channel_bandwidth_B_mhz * 1e6
        T_int_sense = self.uav.config.sensing.t_int_seconds
        P_th_sense = self.uav.config.sensing.p_th_probability
        const_log_term_sense = np.log(1 / (1 - P_th_sense)) if (1 - P_th_sense) > 1e-9 else float('inf')

        for i in range(self.num_areas): # i is the index in the visiting sequence
            actual_area_id = self.fixed_trajectory_solution_for_hover_opt.visiting_sequence_pi[i]
            v_k_fixed = self.fixed_trajectory_solution_for_hover_opt.speeds_v_mps[i+1]

            # t_f^k (use mu_k)
            total_time_expr += mu_k_vars[actual_area_id] / v0_fixed
            # t_D^k (use beta_k)
            total_time_expr += beta_k_vars[actual_area_id] / v_k_fixed
            # t_e^k (use psi_k)
            total_time_expr += T_int_sense * const_log_term_sense * cp.inv_pos(psi_k_vars[actual_area_id])
            # t_s^k (use eta_k)
            total_time_expr += (S_k_bits / B_hz) * cp.inv_pos(eta_k_vars[actual_area_id])

        # t_f^0 (return from p_N to s0)
        # This term depends on p_N, so we handle it as a constant term using p_N^(l) value in SCA iterations
        if self.num_areas > 0:
            last_visited_area_id_in_seq = self.fixed_trajectory_solution_for_hover_opt.visiting_sequence_pi[-1]
            # p_N^(l) is obtained from current_p_all_l_for_tf0
            p_N_l_current = current_p_all_l_for_tf0[last_visited_area_id_in_seq]
            s0_np = self.env.initial_uav_position_s0_meters
            dist_pNl_s0 = np.linalg.norm(p_N_l_current - s0_np)
            time_tf0_l = dist_pNl_s0 / v0_fixed if v0_fixed > 1e-6 else float('inf')
            total_time_expr += time_tf0_l # Add
            
        return total_time_expr

    def solve_p2_2_2_convex_subproblem(self,
                                       constraint_manager: HoveringPositionConstraintManager
                                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], # q_tilde, p
                                                 Optional[np.ndarray], Optional[np.ndarray], # mu, beta
                                                 Optional[np.ndarray], Optional[np.ndarray], # eta, psi
                                                 Optional[float]]: # objective value
        """Solves the P2.2.2 (SCA-approximated P2.2.1) problem using CVXPY."""
        
        # Declare optimization variables
        # For each area k, q̃_k (2D vector) and p_k (2D vector)
        # q_tilde_k_vars[k] is the variable for area_id k
        q_tilde_k_cvx_vars = [cp.Variable(2, name=f"q_tilde_{k}") for k in range(self.num_areas)] # collection position
        p_k_cvx_vars = [cp.Variable(2, name=f"p_{k}") for k in range(self.num_areas)] # transmission position
        
        # Slack variables (scalar values for each area k)
        mu_k_cvx_vars = cp.Variable(self.num_areas, name="mu_k", pos=True)
        beta_k_cvx_vars = cp.Variable(self.num_areas, name="beta_k", pos=True)
        eta_k_cvx_vars = cp.Variable(self.num_areas, name="eta_k", pos=True) # input of log, must be positive
        psi_k_cvx_vars = cp.Variable(self.num_areas, name="psi_k", pos=True) # input of log, must be positive

        # Objective function (use the objective function of P2.2.1)
        objective_expr = self._calculate_total_time_objective_p221_48a(
            mu_k_cvx_vars, beta_k_cvx_vars, eta_k_cvx_vars, psi_k_cvx_vars,
            constraint_manager.current_p_all_l # t_f^0 계산용
        )
        objective = cp.Minimize(objective_expr)

        # Constraints
        constraints = []
        constraints.extend(constraint_manager.get_distance_constraints_mu_beta_cvxpy_48bc(
            q_tilde_k_cvx_vars, p_k_cvx_vars, mu_k_cvx_vars, beta_k_cvx_vars
        ))
        constraints.extend(constraint_manager.get_sca_eta_constraint_cvxpy_51b( # Transmission position is involved, this constraint is problematic
            p_k_cvx_vars, eta_k_cvx_vars
        ))
        constraints.extend(constraint_manager.get_sca_psi_constraint_cvxpy_51c(
            q_tilde_k_cvx_vars, psi_k_cvx_vars
        ))
        constraints.extend(constraint_manager.get_aoi_constraints_slack_cvxpy_48f(
            beta_k_cvx_vars, eta_k_cvx_vars, psi_k_cvx_vars
        ))
        constraints.append(constraint_manager.get_energy_constraints_slack_cvxpy_48g(
            mu_k_cvx_vars, beta_k_cvx_vars, eta_k_cvx_vars, psi_k_cvx_vars
        ))
        constraints.extend(constraint_manager.get_hovering_position_constraints_cvxpy_48hi( # This constraint is problematic, collection position is involved
            q_tilde_k_cvx_vars, p_k_cvx_vars
        ))

        problem = cp.Problem(objective, constraints)
        try:
            # ECOS_BB can handle mixed-integer problems, but here we have continuous variables.
            # ECOS, SCS, CLARABEL, etc. can be used.
            problem.solve(solver= cp.CLARABEL)
        except cp.error.SolverError as e:
            logger.warning(f"CVXPY SolverError in P2.2.2: {e}")
            return None, None, None, None, None, None, None
        except Exception as e:
            logger.error(f"Exception in P2.2.2 CVXPY solve: {e}", exc_info=True)
            return None, None, None, None, None, None, None

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Convert CVXPY variables to numpy arrays
            opt_q_tilde_all = np.array([var.value for var in q_tilde_k_cvx_vars])
            opt_p_all = np.array([var.value for var in p_k_cvx_vars])
            
            opt_mu_k = mu_k_cvx_vars.value
            opt_beta_k = beta_k_cvx_vars.value
            opt_eta_k = eta_k_cvx_vars.value
            opt_psi_k = psi_k_cvx_vars.value
            
            optimal_value = problem.value
            return (opt_q_tilde_all, opt_p_all,
                    opt_mu_k, opt_beta_k, opt_eta_k, opt_psi_k,
                    optimal_value)
        else:
            logger.warning(f"P2.2.2 optimization failed or status not optimal: {problem.status}")
            # Check variable values if failed (for debugging)
            # for var in problem.variables():
            #     logger.debug(f"Var {var.name()}: {var.value}")
            return None, None, None, None, None, None, None

    def optimize_hovering_positions(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """Algorithm 3: SCA-based hovering position optimization."""
        logger.print_subheader("Starting Hovering Position Optimization (Algorithm 3 - SCA)")

        current_q_tilde_l = self.fixed_trajectory_solution_for_hover_opt.hover_positions_q_tilde_meters.copy() # collection position
        current_p_l = self.fixed_trajectory_solution_for_hover_opt.hover_positions_p_meters.copy() # transmission

        constraint_mgr_p22 = HoveringPositionConstraintManager(
            self.uav, self.env, self.sim_cfg,
            self.fixed_trajectory_solution_for_hover_opt, # fixed visiting sequence and speed
            current_q_tilde_l, current_p_l
        )

        previous_objective_value = float('inf')

        for l_iter in range(self.max_sca_iterations):
            # logger.info(f"SCA Iteration {l_iter+1}/{self.max_sca_iterations} for hovering position optimization...")
            
            constraint_mgr_p22.update_current_hovering_positions_for_sca(current_q_tilde_l, current_p_l)

            results = self.solve_p2_2_2_convex_subproblem(constraint_mgr_p22)
            
            opt_q_lp1, opt_p_lp1, _, _, _, _, obj_val_lp1 = results

            if opt_q_lp1 is None or opt_p_lp1 is None or obj_val_lp1 is None:
                logger.warning(f"SCA iteration {l_iter+1} (hover_pos) failed. Returning last successful values if any.")
                if l_iter > 0:
                    return current_q_tilde_l, current_p_l, previous_objective_value
                else:
                    # Initialization failed
                    logger.error("Initial SCA iteration for hovering positions failed.")
                    return None, None, None
            
            logger.debug(f"Iter {l_iter+1} (hover_pos): Objective = {obj_val_lp1:.4f}")

            if abs(obj_val_lp1 - previous_objective_value) < self.sca_convergence_tol and l_iter > 0:
                logger.info(f"SCA (hover_pos) converged at iteration {l_iter+1}.")
                current_q_tilde_l = opt_q_lp1
                current_p_l = opt_p_lp1
                previous_objective_value = obj_val_lp1
                break
            
            current_q_tilde_l = opt_q_lp1
            current_p_l = opt_p_lp1
            previous_objective_value = obj_val_lp1

            logger.info(f"[Hovering] SCA Iter {l_iter+1}: Objective = {obj_val_lp1:.4f}")

            if l_iter == self.max_sca_iterations - 1:
                logger.info("SCA (hover_pos) reached max iterations.")
        
        # Return the final results (q_tilde_all, p_all, final objective value)
        return current_q_tilde_l, current_p_l, previous_objective_value