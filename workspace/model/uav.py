# model/uav.py
import numpy as np
import math
from typing import Tuple

# 필요한 클래스 및 데이터 클래스 임포트
from .environment import Environment
from .trajectory import TrajectorySolution # 직접 사용은 안하지만, 이 클래스의 메서드가 TrajectorySolution을 다룸
from configs.data_classes import UAVConfig
from utils.unit_converter import UnitConverter # 단위 변환 유틸리티

class UAV:
    """
    Define the characteristics and operation model of a UAV (Unmanned Aerial Vehicle).
    Includes calculation logic related to energy consumption, communication, sensing, etc.
    """
    config: UAVConfig
    env: Environment
    # values that can be pre-calculated
    _hovering_power_watts: float
    _noise_power_watts: float
    _snr_gap_linear: float
    _channel_bandwidth_hz: float
    _data_packet_size_bits: float
    _max_energy_joules: float

    def __init__(self, config: UAVConfig, environment: Environment):
        self.config = config
        self.env = environment

        # pre-calculate values that are frequently used and convert units
        self._hovering_power_watts = self.calculate_propulsion_power_watts_25(0.0)
        self._noise_power_watts = UnitConverter.dbm_to_watts(config.communication.noise_power_sigma2_dbm)
        self._snr_gap_linear = UnitConverter.db_to_linear(config.communication.snr_gap_gamma_db)
        self._channel_bandwidth_hz = UnitConverter.mhz_to_hz(config.communication.channel_bandwidth_B_mhz)
        self._data_packet_size_bits = UnitConverter.mbits_to_bits(config.communication.data_packet_size_Sk_mbits)
        self._max_energy_joules = config.energy.max_elimit_joule # 이미 Joule 단위라고 가정


    def get_max_energy_kjoules(self) -> float:
        return self._max_energy_joules / 1000.0

    # --- Propulsion and Energy Methods ---
    def calculate_propulsion_power_watts_25(self, speed_v_mps: float) -> float:
        """
        Calculate the propulsion power consumption (Watts) at a given flight speed.
        P(v) (equation 25)
        """
        P0 = self.config.energy.propulsion_power_model.P0_watts
        P1 = self.config.energy.propulsion_power_model.P1_watts
        v_tip = self.config.energy.propulsion_power_model.v_tip_mps
        v0_hover = self.config.energy.propulsion_power_model.v0_hover_mps # v_0 (bar) in the paper
        d0 = self.config.energy.propulsion_power_model.d0_ratio
        rho = self.config.energy.propulsion_power_model.rho_kg_per_m3
        s_solidity = self.config.energy.propulsion_power_model.s_solidity
        A_disc = self.config.energy.propulsion_power_model.A_disc_m2

        if speed_v_mps < 1e-6: # almost 0, consider hovering
            return P0 + P1

        term1 = P0 * (1 + (3 * speed_v_mps**2) / (v_tip**2))
        
        sqrt_inner_term = (speed_v_mps**4) / (4 * v0_hover**4)
        term2_sqrt_val = math.sqrt(1 + sqrt_inner_term) if (1 + sqrt_inner_term) >=0 else 0 # avoid negative square root
        
        term2 = P1 * (term2_sqrt_val - (speed_v_mps**2) / (2 * v0_hover**2))**(1/2) if (term2_sqrt_val - (speed_v_mps**2) / (2 * v0_hover**2)) >=0 else 0

        term3 = 0.5 * d0 * rho * s_solidity * A_disc * speed_v_mps**3
        
        power = term1 + term2 + term3
        return power

    def get_hovering_power_watts(self) -> float:
        """
        Return the propulsion power consumption (Watts) when the UAV is hovering.
        P_h = P(0)
        """
        return self._hovering_power_watts

    # --- Data Collection Methods ---
    def calculate_data_collection_time_seconds_4(self, hover_q_tilde_k_meters: np.ndarray, area_center_wk_meters: np.ndarray) -> float:
        """
        Calculate the minimum time (t_e^k) required for successful data collection in a specific monitoring area.
        t_e^k (equation 4)
        """
        # In the paper Table I, some values are fixed. Here, we implement equation (4).
        zeta = self.config.sensing.zeta_parameter
        p_th = self.config.sensing.p_th_probability
        t_int = self.config.sensing.t_int_seconds
        H_U = self.config.altitude_hu_meters
    
        # calculate 3D distance d(q̃_k)
        distance_3d_sq = np.sum((hover_q_tilde_k_meters - area_center_wk_meters)**2) + H_U**2
        distance_3d = math.sqrt(distance_3d_sq)

        # calculate numerator/denominator of equation (4)
        numerator = math.log(1 - p_th)
        
        exp_term = math.exp(-zeta * distance_3d)
        if abs(1 - exp_term) < 1e-9: # denominator is close to 0 (exp_term is close to 1)
            return float('inf') # very long time or impossible
        
        denominator = math.log(1 - exp_term)
        if abs(denominator) < 1e-9: # denominator is 0
             return float('inf')

        t_e_k = t_int * (numerator / denominator)
        return max(0, t_e_k) # time cannot be negative

    # --- Data Transmission Methods ---
    def _calculate_uav_gbs_3d_distance_meters_5(self, hover_p_k_meters: np.ndarray) -> float:
        """
        Calculate the 3D distance between the UAV's data transmission hovering position p_k and the GBS.
        d(p_k) (square root of equation 5)
        """
        H_U = self.config.altitude_hu_meters
        H_B = self.env.gbs.altitude_hb_meters # get altitude of GBS from Environment object
        
        # assume the horizontal position of GBS is (0,0)
        horizontal_dist_sq = np.sum(hover_p_k_meters**2)
        vertical_dist_sq = (H_U - H_B)**2
        distance_3d = math.sqrt(horizontal_dist_sq + vertical_dist_sq)
        return distance_3d

    def _calculate_average_channel_power_gain_8(self, hover_p_k_meters: np.ndarray) -> float:
        """
        Calculate the average channel power gain between the UAV and the GBS (linear value).
        E[h(p_k)] (equation 8)
        """
        H_U = self.config.altitude_hu_meters
        H_B = self.env.gbs.altitude_hb_meters
        
        # 1. calculate 3D distance d(p_k) (equation 5)
        distance_3d = self._calculate_uav_gbs_3d_distance_meters_5(hover_p_k_meters)

        if distance_3d < 1e-9: # distance is close to 0 (very close)
            # physically, distance of 0 is impossible, and the channel model is not defined.
            # avoid this case by d_min constraint.
            # here, assume a very large gain or error handling.
            # in the paper, d_min constraint excludes too close cases.
            return float('inf') # or a very large constant value

        # 2. calculate elevation angle θ(p_k) (in degrees)
        # θ(p_k) = (180/π) * arcsin((H_U - H_B) / d(p_k))
        delta_H = H_U - H_B
        
        # the argument of arcsin must be between -1 and 1. due to floating point error, it may exceed 1 slightly.
        # if the UAV is at the same or lower altitude than the GBS, delta_H is 0 or negative.
        # in this case, the elevation angle is 0 or negative.
        if distance_3d == 0: # avoid division by 0
            elevation_angle_rad = math.pi / 2 if delta_H > 0 else 0 # 90 degrees if the same position, 0 degrees otherwise
        else:
            sin_arg = delta_H / distance_3d
            sin_arg = np.clip(sin_arg, -1.0, 1.0) # clip between -1 and 1
            elevation_angle_rad = math.asin(sin_arg)
        
        elevation_angle_deg = math.degrees(elevation_angle_rad)

        # 3. calculate LoS probability P_LoS (equation 9)
        a_los = self.config.communication.los_params.a
        b_los = self.config.communication.los_params.b
        
        # exp(-b * (θ(p_k) - a))
        exp_term_exponent = -b_los * (elevation_angle_deg - a_los)
        # if exp_term_exponent is too large, math.exp may cause OverflowError.
        # especially when b_los is positive and (elevation_angle_deg - a_los) is very negative.
        # in this case, exp_term approaches 0 and P_LoS approaches 1.
        # or if exp_term_exponent is too small, exp_term approaches infinity.
        # in this case, P_LoS approaches 0.
        # to avoid this, clip or conditional processing.
        if exp_term_exponent > 700: # e^700 is approximately 10^304, which may cause overflow
            exp_val = float('inf')
        elif exp_term_exponent < -700: # e^-700 is close to 0
            exp_val = 0.0
        else:
            exp_val = math.exp(exp_term_exponent)

        P_LoS = 1 / (1 + a_los * exp_val)
        P_LoS = np.clip(P_LoS, 0.0, 1.0) # clip between 0 and 1

        # 4. get channel parameters
        beta0_linear = UnitConverter.db_to_linear(self.config.communication.beta0_db)
        alpha = self.config.communication.path_loss_alpha
        kappa_nlos = self.config.communication.kappa_nlos

        # 5. calculate path loss term (β_0 * d(p_k)^(-α))
        path_loss_component = beta0_linear * (distance_3d**(-alpha))

        # 6. calculate average channel gain (equation 8)
        E_h_pk = P_LoS * path_loss_component + (1 - P_LoS) * kappa_nlos * path_loss_component
        
        return E_h_pk


    def calculate_snr_at_gbs_linear_11(self, hover_p_k_meters: np.ndarray) -> float:
        """
        Calculate the average SNR (linear value) at the GBS for a given horizontal distance.
        E[SNR(p_k)] (approximate equation 11)
        """
        Pu_watts = UnitConverter.dbm_to_watts(self.config.communication.pu_transmit_power_dbm)
        beta0_linear = UnitConverter.db_to_linear(self.config.communication.beta0_db)
        # P_LoS_bar: in the paper, it is assumed to be a value for a specific angle, or the average of equation (9).
        # here, we directly get the P_LoS_bar value from config (plos_approx_probability)
        P_LoS_bar = self.config.communication.plos_approx_probability # similar to (1-kappa)*P_LoS + kappa*P_NLoS
                                                                    # P_LoS_bar in equation (11) is of the form (P_LoS + (1-P_LoS)kappa)
                                                                    # here, we use the plos_approx_probability value from config
        
        alpha = self.config.communication.path_loss_alpha
        distance_3d = self._calculate_uav_gbs_3d_distance_meters_5(hover_p_k_meters)
        if distance_3d < 1e-6:
            # when the distance is very close, SNR approaches infinity.
            # in reality, the SNR may actually decrease due to near-field effects of antennas.
            # here, we limit the SNR to a very large value or avoid this case by d_min constraint.
            return 1e10 # example: very large SNR

        # y0 = Pu * beta0 * P_LoS_bar / (sigma^2 * Gamma) (equation 11 numerator)
        # here, P_LoS_bar is of the form (P_LoS(pk) + (1-P_LoS(pk))kappa).
        # we use the plos_approx_probability value from config.
        y0_numerator = Pu_watts * beta0_linear * P_LoS_bar
        y0_denominator = self._noise_power_watts * self._snr_gap_linear
        if abs(y0_denominator) < 1e-20: # avoid division by 0
            return 0.0
        y0 = y0_numerator / y0_denominator
        
        snr_approx = y0 / (distance_3d**alpha)
        return max(0, snr_approx) # SNR cannot be negative

    def calculate_achievable_data_rate_bps_16(self, hover_p_k_meters: np.ndarray) -> float:
        """
        Calculate the achievable average data rate (bps) at a given horizontal distance.
        E[R(p_k)] (approximate equation 16)
        """
        snr_linear = self.calculate_snr_at_gbs_linear_11(hover_p_k_meters)
        
        # B * log2(1 + SNR)
        rate_bps = self._channel_bandwidth_hz * math.log2(1 + snr_linear)
        return max(0, rate_bps) # rate cannot be negative

    def calculate_data_transmission_time_seconds_23(self, hover_p_k_meters: np.ndarray) -> float:
        """
        Calculate the time (t_s^k) required to transmit data of a given size from a specific position to the GBS.
        t_s^k = S_k / E[R(p_k)] (approximate equation 23)
        """
        achievable_rate_bps = self.calculate_achievable_data_rate_bps_16(hover_p_k_meters)
        # print(f"achievable_rate_bps: {achievable_rate_bps}")

        if achievable_rate_bps < 1e-6: # when the rate is close to 0, the transmission time approaches infinity
            return float('inf')
            
        t_s_k = self._data_packet_size_bits / achievable_rate_bps
        return t_s_k

    def get_max_communication_horizontal_distance_dmax_meters_13(self) -> float:
        """
        Calculate the maximum allowable horizontal distance (d_max) between the UAV and the GBS,
        satisfying the minimum SNR requirement.
        d_max (equation 13)
        """
        Pu_watts = UnitConverter.dbm_to_watts(self.config.communication.pu_transmit_power_dbm)
        beta0_linear = UnitConverter.db_to_linear(self.config.communication.beta0_db)
        P_LoS_bar = self.config.communication.plos_approx_probability
        alpha = self.config.communication.path_loss_alpha
        snr_min_linear = UnitConverter.db_to_linear(self.config.communication.snr_min_db)
        H_U = self.config.altitude_hu_meters
        H_B = self.env.gbs.altitude_hb_meters

        # calculate y0 (equation 11 numerator)
        y0_numerator = Pu_watts * beta0_linear * P_LoS_bar
        y0_denominator = self._noise_power_watts * self._snr_gap_linear

        if abs(y0_denominator) < 1e-20: return 0.0 # avoid division by 0
        y0 = y0_numerator / y0_denominator

        if snr_min_linear < 1e-9 : return float('inf') # if SNR_min is close to 0, d_max approaches infinity

        # (y0 / SNR_min)^(2/alpha)
        term1 = (y0 / snr_min_linear)**(2 / alpha)
        term2 = (H_U - H_B)**2
        
        if term1 < term2: # if the square root inner term is negative (physically, d_max does not exist)
            return 0.0 
            
        d_max_horizontal = math.sqrt(term1 - term2)
        return d_max_horizontal

    def get_min_communication_horizontal_distance_dmin_meters_14(self) -> float:
        """
        Calculate the minimum allowable horizontal distance (d_min) between the UAV and the GBS.
        d_min (equation 14)
        This is usually obtained from config as a fixed value.
        """
        return self.config.communication.d_min_comm_meters

    # --- AoI Related Methods ---
    def calculate_max_aoi_for_area_seconds_20(self,
                                              time_data_collection_seconds: float,
                                              time_fly_q_to_p_seconds: float,
                                              time_data_transmission_seconds: float) -> float:
        """
        Calculate the maximum AoI (AoI_max^k) for a specific monitoring area.
        AoI_max^k = T_c^k - T_s^k (equation 20)
        This is equal to t_e^k + (flying time from q̃_k to p_k) + t_s^k.
        """
        return time_data_collection_seconds + time_fly_q_to_p_seconds + time_data_transmission_seconds

    # --- Trajectory Evaluation Method ---
    def calculate_total_mission_time_energy_aois_for_trajectory(self, trajectory: TrajectorySolution) -> Tuple[float, float, np.ndarray]:
        """
        Calculate the total mission completion time (T), total energy consumption (E_all),
        and maximum AoI (AoI_max^k) for a given TrajectorySolution.
        T (equation 22), E_all (equation 28), AoI_max^k (equation 20)
        """
        total_time_T_seconds = 0.0
        flying_energy_Ep_joules = 0.0
        hovering_energy_Eh_joules = 0.0
        
        num_actual_areas_visited = len(trajectory.visiting_sequence_pi)
        if num_actual_areas_visited == 0:
            return 0.0, 0.0, np.array([])

        aois_per_area_seconds = np.zeros(num_actual_areas_visited)

        # UAV speed: v0 is trajectory.speeds_v_mps[0]
        # vk is trajectory.speeds_v_mps[i+1] (i is the visiting sequence index)
        v0_speed_mps = trajectory.speeds_v_mps[0]

        # 1. fly from initial position to the first data collection position q̃_{π(1)}
        current_pos_meters = self.env.initial_uav_position_s0_meters
        first_area_id_in_seq = trajectory.visiting_sequence_pi[0]
        q_tilde_first = trajectory.hover_positions_q_tilde_meters[first_area_id_in_seq]
        
        distance_to_first_q = np.linalg.norm(q_tilde_first - current_pos_meters)
        time_to_first_q = distance_to_first_q / v0_speed_mps if v0_speed_mps > 1e-6 else float('inf')
        
        total_time_T_seconds += time_to_first_q
        flying_energy_Ep_joules += self.calculate_propulsion_power_watts_25(v0_speed_mps) * time_to_first_q
        current_pos_meters = q_tilde_first # update current position

        for i in range(num_actual_areas_visited):
            area_id = trajectory.visiting_sequence_pi[i]
            area_center_wk = self.env.get_area_by_id(area_id).position_wk_meters
            q_tilde_k = trajectory.hover_positions_q_tilde_meters[area_id]
            p_k = trajectory.hover_positions_p_meters[area_id]
            # vk is the speed corresponding to the current area_id.
            # according to the indexing rule of speeds_v_mps.
            # if speeds_v_mps is [v0, v_area0, v_area1, ...],
            # vk_speed_mps = trajectory.speeds_v_mps[area_id + 1] (if area_id starts from 0)
            # or the speed according to the visiting sequence, trajectory.speeds_v_mps[i + 1]
            # here, v_k is interpreted as the speed of "flying from A_k to the transmission position with collected data".
            # that is, used when flying from q_tilde_k to p_k.
            # when flying from previous p to current q, v0 is used.
            vk_speed_mps = trajectory.speeds_v_mps[i + 1] # assume: speeds_v_mps = [v0, v_pi(1), v_pi(2), ...]

            # 2. data collection time t_e^k
            # t_e_k = self.config.sensing.data_collection_time_te_seconds # fixed value is used
            t_e_k = self.calculate_data_collection_time_seconds_4(q_tilde_k, area_center_wk)
            total_time_T_seconds += t_e_k
            hovering_energy_Eh_joules += self._hovering_power_watts * t_e_k

            # 3. fly from q̃_k to p_k (use speed v_k)
            distance_q_to_p = np.linalg.norm(p_k - q_tilde_k)
            time_fly_q_to_p = distance_q_to_p / vk_speed_mps if vk_speed_mps > 1e-6 else float('inf')
            total_time_T_seconds += time_fly_q_to_p
            flying_energy_Ep_joules += self.calculate_propulsion_power_watts_25(vk_speed_mps) * time_fly_q_to_p
            current_pos_meters = p_k # update current position

            # 4. data transmission time t_s^k
            t_s_k = self.calculate_data_transmission_time_seconds_23(p_k)
            total_time_T_seconds += t_s_k
            hovering_energy_Eh_joules += self._hovering_power_watts * t_s_k
            
            # 5. calculate AoI_max^k for the current area
            aois_per_area_seconds[area_id] = self.calculate_max_aoi_for_area_seconds_20(
                t_e_k, time_fly_q_to_p, t_s_k
            )

            # 6. fly from current p_k to the next data collection position q̃_{π(i+1)} (use speed v_0)
            if i < num_actual_areas_visited - 1:
                next_area_id_in_seq = trajectory.visiting_sequence_pi[i+1]
                q_tilde_next = trajectory.hover_positions_q_tilde_meters[next_area_id_in_seq]
                
                distance_p_to_next_q = np.linalg.norm(q_tilde_next - current_pos_meters)
                time_p_to_next_q = distance_p_to_next_q / v0_speed_mps if v0_speed_mps > 1e-6 else float('inf')
                
                total_time_T_seconds += time_p_to_next_q
                flying_energy_Ep_joules += self.calculate_propulsion_power_watts_25(v0_speed_mps) * time_p_to_next_q
                current_pos_meters = q_tilde_next # update current position
        
        # 7. fly from the last p_N to the initial position s_0 (use speed v_0)
        # current_pos_meters is the last p_k position
        distance_last_p_to_initial = np.linalg.norm(self.env.initial_uav_position_s0_meters - current_pos_meters)
        time_last_p_to_initial = distance_last_p_to_initial / v0_speed_mps if v0_speed_mps > 1e-6 else float('inf')
        
        total_time_T_seconds += time_last_p_to_initial # t_F^0 (equation 24)
        flying_energy_Ep_joules += self.calculate_propulsion_power_watts_25(v0_speed_mps) * time_last_p_to_initial

        total_energy_E_all_joules = flying_energy_Ep_joules + hovering_energy_Eh_joules
        
        return total_time_T_seconds, total_energy_E_all_joules, aois_per_area_seconds