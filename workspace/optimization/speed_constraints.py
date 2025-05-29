# workspace/optimization/speed_constraints.py
import numpy as np
import math
import cvxpy as cp # CVXPY 임포트
from typing import Tuple, List, Optional

from model.uav import UAV
from model.environment import Environment
from model.trajectory import TrajectorySolution
from configs.data_classes import SimulationConfig, PropulsionPowerModelParams

class SpeedOptimizationConstraintManager:
    uav: UAV
    env: Environment
    sim_cfg: SimulationConfig
    fixed_path_sequence: List[int]
    fixed_hover_q_all: np.ndarray
    fixed_hover_p_all: np.ndarray

    current_speeds_v_mps_for_sca_iteration_l: np.ndarray # SCA iteration v_m^(l)
    _uav_energy_params: PropulsionPowerModelParams
    _uav_hover_power: float

    # TrajectorySolution object for storing fixed path information
    _fixed_trajectory_path_details: TrajectorySolution

    def __init__(self,
                 uav_model: UAV,
                 env_model: Environment,
                 sim_config: SimulationConfig,
                 fixed_trajectory_path: TrajectorySolution, # only use path information
                 initial_speeds_for_sca: np.ndarray,
                 initial_lambdas_for_sca: np.ndarray): # v_m^(0)
        self.uav = uav_model
        self.env = env_model
        self.sim_cfg = sim_config
        
        # only store fixed path information
        self.fixed_path_sequence = fixed_trajectory_path.visiting_sequence_pi
        self.fixed_hover_q_all = fixed_trajectory_path.hover_positions_q_tilde_meters
        self.fixed_hover_p_all = fixed_trajectory_path.hover_positions_p_meters
        # also store TrajectorySolution object for get_path_segment_details etc.
        self._fixed_trajectory_path_details = fixed_trajectory_path

        self.current_speeds_v_mps_for_sca_iteration_l = initial_speeds_for_sca.copy()
        self.current_lambdas_for_sca_iteration_l = initial_lambdas_for_sca.copy()
        self._uav_energy_params = self.uav.config.energy.propulsion_power_model
        self._uav_hover_power = self.uav.get_hovering_power_watts()

        self.epsilon = 1e-6 # small value for numerical stability

    def update_current_speeds_for_sca_iteration(self, new_speeds_v_mps: np.ndarray):
        self.current_speeds_v_mps_for_sca_iteration_l = new_speeds_v_mps.copy()

    def update_current_lambdas_for_sca_iteration(self, new_lambdas_for_sca: np.ndarray):
        self.current_lambdas_for_sca_iteration_l = new_lambdas_for_sca.copy()

    # --- CVXPY constraint generation methods ---
    def get_aoi_constraints_cvxpy_30b(self, v_m_vars: cp.Variable) -> List[cp.Expression]:
        """Returns a list of CVXPY constraints for equation (30b)."""
        constraints = []
        aoi_limit_val = self.sim_cfg.general.aoi_limit_seconds
        num_visited_areas = len(self.fixed_path_sequence)

        for i in range(num_visited_areas):
            actual_area_id = self.fixed_path_sequence[i]
            q_k = self.fixed_hover_q_all[actual_area_id]
            p_k = self.fixed_hover_p_all[actual_area_id]

            area_center_wk = self.env.get_area_by_id(actual_area_id).position_wk_meters

            const_t_e_k = self.uav.calculate_data_collection_time_seconds_4(q_k, area_center_wk)
            const_t_s_k = self.uav.calculate_data_transmission_time_seconds_23(p_k)
            const_dist_q_to_p = np.linalg.norm(p_k - q_k)
            
            v_k_var_for_aoi = v_m_vars[i+1] # v_m_vars[0] is v0, v_m_vars[1] is v1, etc.
            
            constraints.append(
                const_t_e_k + const_dist_q_to_p * cp.inv_pos(v_k_var_for_aoi) + const_t_s_k <= aoi_limit_val
            )
        return constraints

    def get_speed_limit_constraints_cvxpy_30c(self, v_m_vars: cp.Variable) -> List[cp.Expression]:
        """Returns a list of CVXPY constraints for equation (30c)."""
        constraints = []
        v_max_limit = self.uav.config.speed.max_vmax_mps
        constraints.append(v_m_vars >= 1e-6) # slightly larger than 0 (pos=True is used)
        constraints.append(v_m_vars <= v_max_limit)
        return constraints

    def get_energy_constraint_sca_cvxpy_34b(self, v_m_vars: cp.Variable, lambda_m_vars: cp.Variable) -> cp.Expression:
        """Returns a CVXPY expression for the energy constraint (34b)."""
        total_energy_sca_expr = 0.0
        # use _fixed_trajectory_path_details to get path segment information
        path_segments_for_energy = self._fixed_trajectory_path_details.get_path_segment_details(self.env)
        
        P0 = self._uav_energy_params.P0_watts
        P1 = self._uav_energy_params.P1_watts
        v_tip = self._uav_energy_params.v_tip_mps
        d0_cfg = self._uav_energy_params.d0_ratio
        rho_cfg = self._uav_energy_params.rho_kg_per_m3
        s_solidity_cfg = self._uav_energy_params.s_solidity
        A_disc_cfg = self._uav_energy_params.A_disc_m2

        for segment in path_segments_for_energy:
            D = segment['distance']
            v_var = v_m_vars[segment['speed_idx']]
            lambda_var = lambda_m_vars[segment['slack_idx']]
            term_P0_inv = P0 * D * cp.inv_pos(v_var)
            term_P0_lin = P0 * 3 * D / (v_tip**2) * v_var
            term_P1_lin = P1 * D * lambda_var
            term_drag_quad = 0.5 * d0_cfg * rho_cfg * s_solidity_cfg * A_disc_cfg * D * cp.power(v_var, 2)
            total_energy_sca_expr += term_P0_inv + term_P0_lin + term_P1_lin + term_drag_quad
        
        total_hovering_time_fixed_for_energy = 0.0
        num_visited_areas = len(self.fixed_path_sequence)
        for i in range(num_visited_areas):
            area_id = self.fixed_path_sequence[i]
            q_k_fixed = self.fixed_hover_q_all[area_id]
            p_k_fixed = self.fixed_hover_p_all[area_id]
            area_center_fixed = self.env.get_area_by_id(area_id).position_wk_meters
            total_hovering_time_fixed_for_energy += self.uav.calculate_data_collection_time_seconds_4(q_k_fixed, area_center_fixed)
            total_hovering_time_fixed_for_energy += self.uav.calculate_data_transmission_time_seconds_23(p_k_fixed)

        total_energy_sca_expr += self._uav_hover_power * total_hovering_time_fixed_for_energy
        return total_energy_sca_expr <= self.uav.config.energy.max_elimit_joule

    def get_sca_slack_constraints_cvxpy_36b_34d(self, v_m_vars: cp.Variable, lambda_m_vars: cp.Variable) -> List[cp.Expression]:
        """
        Assumption: the paper derives a convex constraint of the form f_ub(lambda_m) <= g_lb(v_m) from the inequality 1 / (lambda_m^4 + lambda_m^2/v0_bar^2) <= v_m^4 (equation 34c)
        Left-hand side f(lambda_m) = 1 / (lambda_m^4 + lambda_m^2/v0_bar^2) is Taylor expanded around lambda_m^(l) (upper approximation)
        Right-hand side g(v_m) = v_m^4 is Taylor expanded around v_m^(l) (lower approximation)
        So, f_ub(lambda_m) <= g_lb(v_m) is derived.
        
        f(lambda_m) = (lambda_m^4 + lambda_m^2/v0_bar^2)^(-1)
        f'(lambda_m) = -1 * (lambda_m^4 + lambda_m^2/v0_bar^2)^(-2) * (4*lambda_m^3 + 2*lambda_m/v0_bar^2)
        
        f(lambda_m) is concave over lambda_m > 0.
        If it is concave, f_ub(lambda_m) = f(lambda_m^(l)) + f'(lambda_m^(l))*(lambda_m - lambda_m^(l))
        """
        constraints = []
        v0_bar_sq = self._uav_energy_params.v0_hover_mps**2
        if v0_bar_sq < self.epsilon: v0_bar_sq = self.epsilon

        for i in range(len(self.current_speeds_v_mps_for_sca_iteration_l)):
            v_m_l_val = self.current_speeds_v_mps_for_sca_iteration_l[i]
            lambda_m_l_val = self.current_lambdas_for_sca_iteration_l[i]

            v_m_var = v_m_vars[i]
            lambda_m_var = lambda_m_vars[i]

            # Right-hand side g(v_m) = v_m^4 is lower approximation g_lb(v_m) (equation 35)
            safe_v_m_l_val = np.maximum(self.epsilon, v_m_l_val)
            g_lb_v_expr = (safe_v_m_l_val**4) + 4 * (safe_v_m_l_val**3) * (v_m_var - safe_v_m_l_val)
            # g_lb_v_expr is an affine function, so cp.maximum is not needed (but v_m_var >= epsilon constraint is needed)

            # Left-hand side f(lambda_m) = 1 / (lambda_m^4 + lambda_m^2/v0_bar^2) is upper approximation f_ub(lambda_m)
            safe_lambda_m_l_val = np.maximum(self.epsilon, lambda_m_l_val)
            
            # h(lambda_m) = lambda_m^4 + lambda_m^2/v0_bar^2
            h_lambda_l = safe_lambda_m_l_val**4 + (safe_lambda_m_l_val**2) / v0_bar_sq
            if h_lambda_l < self.epsilon: h_lambda_l = self.epsilon # avoid division by zero

            f_lambda_l = 1.0 / h_lambda_l
            
            # h'(lambda_m) = 4*lambda_m^3 + 2*lambda_m/v0_bar^2
            h_prime_lambda_l = 4 * safe_lambda_m_l_val**3 + (2 * safe_lambda_m_l_val) / v0_bar_sq
            
            # f'(lambda_m) = -h'(lambda_m) / h(lambda_m)^2
            f_prime_lambda_l = -h_prime_lambda_l / (h_lambda_l**2)
            
            f_ub_lambda_expr = f_lambda_l + f_prime_lambda_l * (lambda_m_var - safe_lambda_m_l_val)
            # f_ub_lambda_expr is an affine function

            # Constraint: f_ub(lambda_m) <= g_lb(v_m)
            constraints.append(f_ub_lambda_expr <= g_lb_v_expr)
            
            constraints.append(lambda_m_var >= self.epsilon)
            # v_m_var >= self.epsilon is handled in get_speed_limit

        return constraints