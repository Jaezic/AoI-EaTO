# workspace/optimization/hovering_pos_constraints.py
import numpy as np
import math
import cvxpy as cp
from typing import List, Tuple, Optional

from model.uav import UAV
from model.environment import Environment
from model.trajectory import TrajectorySolution
from configs.data_classes import SimulationConfig, UAVConfig

class HoveringPositionConstraintManager:
    uav: UAV
    env: Environment
    sim_cfg: SimulationConfig
    
    # In P2.2, the visiting sequence and speed are fixed
    fixed_visiting_sequence_pi: List[int]
    fixed_speeds_v_mps: np.ndarray

    # Current hovering positions q_k^(l), p_k^(l) during SCA iterations
    current_q_tilde_all_l: np.ndarray
    current_p_all_l: np.ndarray
    
    num_areas: int

    def __init__(self,
                 uav_model: UAV,
                 env_model: Environment,
                 sim_config: SimulationConfig,
                 fixed_trajectory_solution_for_hover_opt: TrajectorySolution, # 방문 순서, 속도 정보 사용
                 initial_q_tilde_all_for_sca: np.ndarray, # q_k^(0)
                 initial_p_all_for_sca: np.ndarray):      # p_k^(0)
        self.uav = uav_model
        self.env = env_model
        self.sim_cfg = sim_config

        self.fixed_visiting_sequence_pi = fixed_trajectory_solution_for_hover_opt.visiting_sequence_pi
        self.fixed_speeds_v_mps = fixed_trajectory_solution_for_hover_opt.speeds_v_mps
        
        self.current_q_tilde_all_l = initial_q_tilde_all_for_sca.copy()
        self.current_p_all_l = initial_p_all_for_sca.copy()
        
        self.num_areas = env_model.get_total_monitoring_areas()
        if len(self.fixed_visiting_sequence_pi) != self.num_areas:
            # In general, the visiting sequence includes all areas, but partial visiting scenarios are also possible
            # Here, we assume all areas are visited
            pass

    def update_current_hovering_positions_for_sca(self,
                                                  new_q_tilde_all: np.ndarray,
                                                  new_p_all: np.ndarray):
        self.current_q_tilde_all_l = new_q_tilde_all.copy()
        self.current_p_all_l = new_p_all.copy()

    # --- Constraint generation methods for P2.2.1 and P2.2.2 ---

    def get_distance_constraints_mu_beta_cvxpy_48bc(self,
                                                    q_tilde_k_vars: List[cp.Variable], # 각 지역 k의 q̃_k (2D 벡터)
                                                    p_k_vars: List[cp.Variable],       # 각 지역 k의 p_k (2D 벡터)
                                                    mu_k_vars: cp.Variable,      # 모든 지역의 μ_k (스칼라)
                                                    beta_k_vars: cp.Variable     # 모든 지역의 β_k (스칼라)
                                                   ) -> List[cp.Expression]:
        """Returns a list of CVXPY constraints for equations (48b) and (48c)."""
        constraints = []
        
        # Previous transmission position p̃_{π(π'(k)-1)}
        # For the first area, the previous position is the initial UAV position s0
        # For the last area, there is no next position (or the path to return to s0 is handled separately)
        
        # p_prev_for_mu_k: Previous p position used for each mu_k calculation
        # Based on the visiting sequence, p_k_vars or s0 is used
        p_prev_for_mu_k_list = []
        s0_np = self.env.initial_uav_position_s0_meters
        
        if self.num_areas > 0:
            # First visited area (i=0)
            p_prev_for_mu_k_list.append(s0_np) # Constant value
            
            # From the second to the last visited area
            for i in range(1, self.num_areas):
                prev_visited_area_id_in_seq = self.fixed_visiting_sequence_pi[i-1]
                # Assume p_k_vars uses area_id as index
                p_prev_for_mu_k_list.append(p_k_vars[prev_visited_area_id_in_seq])

        for i in range(self.num_areas): # Index from 0 to N-1 (index of visiting sequence)
            actual_area_id = self.fixed_visiting_sequence_pi[i]
            
            q_k_var_current = q_tilde_k_vars[actual_area_id]
            p_k_var_current = p_k_vars[actual_area_id]
            
            # (48b) ||p̃_{prev} - q̃_k|| <= μ_k
            # mu_k_vars is a variable for all areas, so use mu_k_vars[actual_area_id] for the current area (actual_area_id)
            p_prev_val_or_var = p_prev_for_mu_k_list[i]
            constraints.append(cp.norm(p_prev_val_or_var - q_k_var_current, 2) <= mu_k_vars[actual_area_id])
            # (48c) ||q̃_k - p_k|| <= β_k
            constraints.append(cp.norm(q_k_var_current - p_k_var_current, 2) <= beta_k_vars[actual_area_id])
            
        return constraints

    def get_sca_eta_constraint_cvxpy_51b(self,
                                          p_k_vars: List[cp.Variable], # 각 지역 k의 p_k (2D 벡터)
                                          eta_k_vars: cp.Variable,     # 모든 지역의 η_k (스칼라)
                                         ) -> List[cp.Expression]:
        """Returns a list of CVXPY constraints for equation (51b) (η_k ≤ R_k^(lb))."""
        constraints = []
        H_U = self.uav.config.altitude_hu_meters
        H_B = self.env.gbs.altitude_hb_meters
        alpha_pl = self.uav.config.communication.path_loss_alpha
        
        for k_area_id in range(self.num_areas):
            p_k_var     = p_k_vars[k_area_id]          # Variable
            p_k_l       = self.current_p_all_l[k_area_id]  # Fixed point p_k^(l)

            # --- (1) Calculate constant SNR(p_k^(l)) ---
            snr_l = self.uav.calculate_snr_at_gbs_linear_11(p_k_l)
            log_term_val = np.log2(1.0 + snr_l)        # log2(1+SNR)

            # --- (2) Calculate omega_k^(l) ---
            # d/d(||p||^2) log2(1 + y0 * (||p||^2 + ΔH^2)^(-α/2))
            #   = - (α/2) * log2(e) * snr_l / (||p_k^(l)||^2 + ΔH^2) / (1 + snr_l)
            dist_sq_l = np.sum(p_k_l**2) + (H_U - H_B)**2
            omega_k_val = ( -0.5 * alpha_pl * np.log2(np.e) * snr_l
                            / dist_sq_l / (1.0 + snr_l) )

            # Note: To create a lower bound in SCA, we need to use the tangent of a concave function.
            # log2(1+SNR(r)) is a concave function of r^(-α/2) → ω_k is negative.
            # So, it is added as a *sum* instead of a *subtraction*.
            R_lb_expr = (
                log_term_val
                + omega_k_val * (cp.sum_squares(p_k_var) - np.sum(p_k_l**2))
            )

            # --- (3) 51b constraint ---
            constraints.append(eta_k_vars[k_area_id] <= R_lb_expr)
            constraints.append(eta_k_vars[k_area_id] >= 1e-6)  # Positive
    
        return constraints
        
    def get_sca_psi_constraint_cvxpy_51c(self,
                                         q_tilde_k_vars: List[cp.Variable], # q̃_k for each area k (2D vector)
                                         psi_k_vars: cp.Variable,      # ψ_k for all areas (scalar)
                                        ) -> List[cp.Expression]:
        """Returns a list of CVXPY constraints for equation (51c) (ψ_k ≤ Y_k^(lb))."""
        constraints = []
        H_U = self.uav.config.altitude_hu_meters
        zeta_sense = self.uav.config.sensing.zeta_parameter
        
        area_centers_wk = self.env.get_all_area_positions_wk() # (N, 2) array

        for k_area_id in range(self.num_areas): # k_area_id is from 0 to N-1 (actual area ID)
            q_k_var_current = q_tilde_k_vars[k_area_id]
            psi_k_var_current = psi_k_vars[k_area_id]
            
            q_k_l_current = self.current_q_tilde_all_l[k_area_id] # q̃_k^(l)
            w_k_current = area_centers_wk[k_area_id]
            
            # Y_k^(lb) calculation (equation 50)
            # ln(1 / (1 - exp(-zeta * d(q̃_k^(l)))))
            # - phi_k * (||q̃_k - w_k||^2 - ||q̃_k^(l) - w_k||^2)
            # Here, phi_k is the derivative coefficient of equation 50
            
            dist_qkl_wk_sq = np.sum((q_k_l_current - w_k_current)**2)
            d_qkl = np.sqrt(dist_qkl_wk_sq + H_U**2)
            
            exp_term_val = np.exp(-zeta_sense * d_qkl)
            if (1 - exp_term_val) < 1e-9: # Avoid division by 0 or log(negative)
                log_term_val_y = -float('inf')
                phi_k_val = 0
            else:
                log_term_val_y = np.log(1 / (1 - exp_term_val))
                
                # phi_k calculation (equation 50)
                # (zeta * exp(-zeta * d(q̃_k^(l)))) / (2 * d(q̃_k^(l)) * (1 - exp(-zeta * d(q̃_k^(l)))))
                if d_qkl < 1e-9 or (1 - exp_term_val) < 1e-9: # Avoid division by 0
                    phi_k_val = 0
                else:
                    phi_num = zeta_sense * exp_term_val
                    phi_den = 2 * d_qkl * (1 - exp_term_val)
                    phi_k_val = phi_num / phi_den if abs(phi_den) > 1e-12 else 0
            
            # Y_k^(lb) expression
            # ||q̃_k - w_k||^2 is expressed as cp.sum_squares(q_k_var_current - w_k_current)
            Y_k_lb_expr = log_term_val_y - phi_k_val * \
                          (cp.sum_squares(q_k_var_current - w_k_current) - dist_qkl_wk_sq)
            
            constraints.append(psi_k_var_current <= Y_k_lb_expr)
            
        return constraints

    def get_aoi_constraints_slack_cvxpy_48f(self,
                                            beta_k_vars: cp.Variable,
                                            eta_k_vars: cp.Variable,
                                            psi_k_vars: cp.Variable
                                           ) -> List[cp.Expression]:
        """Returns a list of CVXPY constraints for equation (48f) (AoI constraint, using slack variables)."""
        constraints = []
        aoi_limit_val = self.sim_cfg.general.aoi_limit_seconds
        
        # Fixed speed v_k (q_tilde_k -> p_k section)
        # fixed_speeds_v_mps[0] is v0, fixed_speeds_v_mps[i+1] is v_{pi(i)}
        
        # Constants for data collection/transmission
        S_k_bits = self.uav.config.communication.data_packet_size_Sk_mbits * 1e6
        B_hz = self.uav.config.communication.channel_bandwidth_B_mhz * 1e6
        T_int_sense = self.uav.config.sensing.t_int_seconds
        P_th_sense = self.uav.config.sensing.p_th_probability
        const_log_term_sense = np.log(1 / (1 - P_th_sense)) if (1 - P_th_sense) > 1e-9 else float('inf')


        for i in range(self.num_areas): # Index of visiting sequence
            actual_area_id = self.fixed_visiting_sequence_pi[i]
            v_k_fixed = self.fixed_speeds_v_mps[i+1] # v_{pi(i)}
            
            beta_k_var_current = beta_k_vars[actual_area_id]
            eta_k_var_current = eta_k_vars[actual_area_id]
            psi_k_var_current = psi_k_vars[actual_area_id]
            
            # t_e^k = T_int * ln(1/(1-P_th)) / ψ_k
            # t_D^k = β_k / v_k
            # t_s^k = S_k / (B * η_k)
            
            # inv_pos can only be used for positive variables. eta_k, psi_k must be positive.
            # If eta_k, psi_k approach 0, time becomes infinite. Need to add constraints eta_k >= eps, psi_k >= eps.
            
            aoi_expr = (T_int_sense * const_log_term_sense * cp.inv_pos(psi_k_var_current) +
                        beta_k_var_current / v_k_fixed +
                        (S_k_bits / B_hz) * cp.inv_pos(eta_k_var_current))
            
            constraints.append(aoi_expr <= aoi_limit_val)
            constraints.append(eta_k_var_current >= 1e-6) # eta_k > 0 (actually a small positive number)
            constraints.append(psi_k_var_current >= 1e-6) # psi_k > 0
            
        return constraints

    def get_energy_constraints_slack_cvxpy_48g(self,
                                               mu_k_vars: cp.Variable,
                                               beta_k_vars: cp.Variable,
                                               eta_k_vars: cp.Variable,
                                               psi_k_vars: cp.Variable
                                              ) -> cp.Expression:
        """Returns a CVXPY expression for equation (48g) (energy constraint, using slack variables)."""
        total_energy_expr = 0.0
        
        v0_fixed = self.fixed_speeds_v_mps[0]
        P_v0_fixed = self.uav.calculate_propulsion_power_watts_25(v0_fixed)
        P_hover_fixed = self.uav.get_hovering_power_watts()

        # Flying energy (using mu_k, beta_k)
        for i in range(self.num_areas): # Index of visiting sequence
            actual_area_id = self.fixed_visiting_sequence_pi[i]
            v_k_fixed = self.fixed_speeds_v_mps[i+1] # v_{pi(i)}
            P_vk_fixed = self.uav.calculate_propulsion_power_watts_25(v_k_fixed)
            
            # E_flying_mu_k = P(v0) * (mu_k / v0)
            total_energy_expr += P_v0_fixed * (mu_k_vars[actual_area_id] / v0_fixed)
            # E_flying_beta_k = P(vk) * (beta_k / vk)
            total_energy_expr += P_vk_fixed * (beta_k_vars[actual_area_id] / v_k_fixed)

        # Flying energy from the last p_N to s0
        # This part is not included in mu_k, beta_k, so we need to use the distance between the last p_k_vars and s0.
        # However, in P2.2.2, p_k_vars are optimization variables, so this term is non-convex.
        # In the paper, t_f^0 is treated as a constant or included in the objective function of P0.
        # Here, the fixed visiting sequence is given as input for Algorithm 3,
        # so we use the distance between the last p_k_vars and s0 as a constant term,
        # or exclude this term and consider only the remaining energy constraints (equation (48g) in the paper does not explicitly include t_f^0).
        # Here, we calculate the energy excluding t_f^0 (based on equations (26), (27))
        
        # Hovering energy (using eta_k, psi_k)
        S_k_bits = self.uav.config.communication.data_packet_size_Sk_mbits * 1e6
        B_hz = self.uav.config.communication.channel_bandwidth_B_mhz * 1e6
        T_int_sense = self.uav.config.sensing.t_int_seconds
        P_th_sense = self.uav.config.sensing.p_th_probability
        const_log_term_sense = np.log(1 / (1 - P_th_sense)) if (1 - P_th_sense) > 1e-9 else float('inf')

        for k_area_id in range(self.num_areas):
            # E_hover_collect = P_h * t_e^k = P_h * T_int * ln(1/(1-P_th)) / ψ_k
            total_energy_expr += P_hover_fixed * (T_int_sense * const_log_term_sense * cp.inv_pos(psi_k_vars[k_area_id]))
            # E_hover_transmit = P_h * t_s^k = P_h * S_k / (B * η_k)
            total_energy_expr += P_hover_fixed * ((S_k_bits / B_hz) * cp.inv_pos(eta_k_vars[k_area_id]))
            
        # Energy related to t_f^0 (returning from p_N to s0)
        # This part depends on the last value of p_k_vars, so we use the last value of p_k^(l) during
        if self.num_areas > 0:
            last_visited_area_id_in_seq = self.fixed_visiting_sequence_pi[-1]
            p_N_l_current = self.current_p_all_l[last_visited_area_id_in_seq]
            s0_np = self.env.initial_uav_position_s0_meters
            dist_pNl_s0 = np.linalg.norm(p_N_l_current - s0_np)
            time_tf0_l = dist_pNl_s0 / v0_fixed if v0_fixed > 1e-6 else float('inf')
            energy_tf0_l = P_v0_fixed * time_tf0_l
            total_energy_expr += energy_tf0_l # Constant term

        return total_energy_expr <= self.uav.config.energy.max_elimit_joule

    def get_hovering_position_constraints_cvxpy_48hi(self,
                                                     q_tilde_k_vars: List[cp.Variable],
                                                     p_k_vars: List[cp.Variable]
                                                    ) -> List[cp.Expression]:
        """Returns a list of CVXPY constraints for equations (48h), (48i) (original 40d, 40e)."""
        constraints = []
        d_min_comm = self.uav.get_min_communication_horizontal_distance_dmin_meters_14()
        d_max_comm = self.uav.get_max_communication_horizontal_distance_dmax_meters_13()
        
        r_u_sense = self.uav.config.sensing.uav_monitoring_range_ru_meters
        r_a_area = self.env.config.area_radius_ra_meters # Assume all areas have the same radius
        max_dist_q_to_wk = r_u_sense - r_a_area # Maximum distance for collection at the current position
        if max_dist_q_to_wk < 0: max_dist_q_to_wk = 0 # Avoid physically impossible cases

        area_centers_wk = self.env.get_all_area_positions_wk()

        for k_area_id in range(self.num_areas):
            p_k_var_current = p_k_vars[k_area_id]
            q_k_var_current = q_tilde_k_vars[k_area_id]
            w_k_const = area_centers_wk[k_area_id]
            
            # (48h) d_min <= ||p_k|| <= d_max
            norm_p_k = cp.norm(p_k_var_current, 2)
            # constraints.append(norm_p_k >= d_min_comm) # This part is problematic, comment out for now
            constraints.append(norm_p_k <= d_max_comm)
            
            # (48i) ||q̃_k - w_k|| <= r_u - r_a
            constraints.append(cp.norm(q_k_var_current - w_k_const, 2) <= max_dist_q_to_wk)
            
        return constraints


    def get_slack_variable_positivity_constraints(self,
                                                  mu_k_vars: cp.Variable,
                                                  beta_k_vars: cp.Variable,
                                                  eta_k_vars: cp.Variable,
                                                  psi_k_vars: cp.Variable
                                                 ) -> List[cp.Expression]:
        """Adds constraints that the slack variables must be positive (similar to pos=True when declaring CVXPY variables)."""
        constraints = []
        eps = 1e-7 # Very small positive number
        constraints.append(mu_k_vars >= eps)
        constraints.append(beta_k_vars >= eps)
        constraints.append(eta_k_vars >= eps)
        constraints.append(psi_k_vars >= eps)
        return constraints