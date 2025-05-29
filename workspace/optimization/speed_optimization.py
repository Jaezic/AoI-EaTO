# workspace/optimization/speed_optimization.py

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional

from model.uav import UAV
from model.environment import Environment
from model.trajectory import TrajectorySolution
from configs.data_classes import SimulationConfig, UAVConfig
from .speed_constraints import SpeedOptimizationConstraintManager
from utils import data_logger as logger

class SpeedOptimizerSCA:
    """
    Algorithm 1: SCA-Based Speed Optimization for Solving P1.1 (and P1.2 by iteration).
    Given a fixed path (visiting sequence, hovering positions), find the optimal flight speed v_m and
    slack variable lambda_m to minimize the total mission completion time.
    """
    uav: UAV
    env: Environment
    sim_cfg: SimulationConfig
    fixed_trajectory_path: TrajectorySolution # only use path information

    # SCA iteration related parameters
    max_sca_iterations: int
    sca_convergence_tol: float

    def __init__(self,
                 uav_model: UAV,
                 env_model: Environment,
                 sim_config: SimulationConfig,
                 trajectory_solution: TrajectorySolution,
                 max_sca_iterations: int = 10, # number of SCA iterations
                 sca_convergence_tol: float = 1e-4): # SCA convergence tolerance
        self.uav = uav_model
        self.env = env_model
        self.sim_cfg = sim_config
        self.fixed_trajectory_path = trajectory_solution # only store path information
        self.max_sca_iterations = max_sca_iterations
        self.sca_convergence_tol = sca_convergence_tol

    def _calculate_total_time_objective(self,
                                        speeds_v_mps: cp.Variable,
                                       ) -> cp.Expression:
        path_segments = self.fixed_trajectory_path.get_path_segment_details(self.env)
        num_areas = self.fixed_trajectory_path.get_num_visited_areas()
        total_time_expr = 0.0
        for segment in path_segments:
            distance = segment['distance']
            speed_var_for_segment = speeds_v_mps[segment['speed_idx']]
            # if distance > 0.1 :
            #      total_time_expr += distance * cp.inv_pos(speed_var_for_segment)
            # else:
            #     print(f"Distance is too small: {distance}, segment idx: {segment['speed_idx']}")
            total_time_expr += distance * cp.inv_pos(speed_var_for_segment)
        total_hovering_time_fixed = 0.0
        for i in range(num_areas):
            area_id = self.fixed_trajectory_path.visiting_sequence_pi[i]
            q_tilde_k = self.fixed_trajectory_path.get_q_tilde_k_for_area_id(area_id)
            p_k = self.fixed_trajectory_path.get_p_k_for_area_id(area_id)
            area_center_wk = self.env.get_area_by_id(area_id).position_wk_meters
            t_e_k = self.uav.calculate_data_collection_time_seconds_4(q_tilde_k, area_center_wk)
            total_hovering_time_fixed += t_e_k
            t_s_k = self.uav.calculate_data_transmission_time_seconds_23(p_k)
            total_hovering_time_fixed += t_s_k
        total_time_expr += total_hovering_time_fixed
        return total_time_expr


    def solve_p1_2_convex_subproblem(self,
                                     constraint_manager: SpeedOptimizationConstraintManager
                                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        num_speeds = len(constraint_manager.current_speeds_v_mps_for_sca_iteration_l)

        v_m_vars = cp.Variable(num_speeds, name="speeds_v_mps", pos=True)
        lambda_m_vars = cp.Variable(num_speeds, name="slacks_lambda_m", pos=True)

        objective_expr = self._calculate_total_time_objective(v_m_vars)
        objective = cp.Minimize(objective_expr)

        constraints = []

        # use SpeedOptimizationConstraintManager's methods to add constraints
        constraints.extend(constraint_manager.get_aoi_constraints_cvxpy_30b(v_m_vars))
        constraints.extend(constraint_manager.get_speed_limit_constraints_cvxpy_30c(v_m_vars))
        
        # energy constraint is a single expression, so use append instead of extend
        constraints.append(constraint_manager.get_energy_constraint_sca_cvxpy_34b(v_m_vars, lambda_m_vars)) # there is a problem
        
        # slack variable relationship constraints
        constraints.extend(constraint_manager.get_sca_slack_constraints_cvxpy_36b_34d(v_m_vars, lambda_m_vars))
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.CLARABEL) 
        except cp.error.SolverError as e:
            logger.warning(f"CVXPY SolverError in P1.2: {e}")
            return None, None, None
        except Exception as e:
            logger.error(f"Exception in P1.2 CVXPY solve: {e}", exc_info=True)
            return None, None, None

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            optimized_speeds = v_m_vars.value
            optimized_slacks = lambda_m_vars.value
            optimal_value = problem.value
            return optimized_speeds, optimized_slacks, optimal_value
        else:
            logger.warning(f"P1.2 optimization failed or status not optimal: {problem.status}")
            return None, None, None


    def optimize_speeds(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        logger.print_subheader("Starting Speed Optimization (Algorithm 1 - SCA)")

        current_v_m_l = self.fixed_trajectory_path.speeds_v_mps.copy()
        
        if current_v_m_l is None or len(current_v_m_l) == 0:
            num_speeds_needed = self.fixed_trajectory_path.get_num_visited_areas() + 1
            current_v_m_l = np.full(num_speeds_needed, self.uav.config.speed.max_vmax_mps / 2.0)

        current_lambda_m_l = np.ones_like(current_v_m_l, dtype=float) * 0.1 
        
        # initialize SpeedOptimizationConstraintManager with fixed_trajectory_path
        constraint_mgr_p1 = SpeedOptimizationConstraintManager(
            self.uav, self.env, self.sim_cfg, self.fixed_trajectory_path, current_v_m_l, current_lambda_m_l
        )

        previous_objective_value = float('inf')

        for l in range(self.max_sca_iterations):            
            constraint_mgr_p1.update_current_speeds_for_sca_iteration(current_v_m_l)
            constraint_mgr_p1.update_current_lambdas_for_sca_iteration(current_lambda_m_l)

            opt_v_m_lp1, opt_lambda_m_lp1, obj_val_lp1 = self.solve_p1_2_convex_subproblem(constraint_mgr_p1)

            if opt_v_m_lp1 is None or obj_val_lp1 is None:
                logger.warning(f"SCA iteration {l+1} failed to find a solution. Returning last successful values if any.")
                if l > 0 : 
                    return current_v_m_l, current_lambda_m_l, previous_objective_value
                else: 
                    return None, None, None

            logger.debug(f"Iter {l+1}: Objective = {obj_val_lp1:.4f}")

            if abs(obj_val_lp1 - previous_objective_value) < self.sca_convergence_tol and l > 0 :
                logger.info(f"SCA converged at iteration {l+1}.")
                current_v_m_l = opt_v_m_lp1
                current_lambda_m_l = opt_lambda_m_lp1
                previous_objective_value = obj_val_lp1
                break
            
            current_v_m_l = opt_v_m_lp1
            current_lambda_m_l = opt_lambda_m_lp1
            previous_objective_value = obj_val_lp1

            logger.info(f"[Speed] SCA Iter {l+1}: Speed = {current_v_m_l}, Objective = {obj_val_lp1:.4f}")

            if l == self.max_sca_iterations - 1:
                logger.info("SCA reached max iterations for speed optimization.")

        return current_v_m_l, current_lambda_m_l, previous_objective_value