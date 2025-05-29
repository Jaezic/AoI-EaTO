# model/trajectory.py
import numpy as np
from typing import List, Optional, Dict, Any # Dict, Any 추가
from dataclasses import dataclass, field

@dataclass
class TrajectorySolution:
    """
    A data class representing the solution of the UAV trajectory.
    It contains the variables optimized by the optimization algorithm and the calculated performance metrics.
    It corresponds to Ω (set of optimization variables) in the paper.
    """
    visiting_sequence_pi: List[int] = field(default_factory=list)
    speeds_v_mps: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hover_positions_q_tilde_meters: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float))
    hover_positions_p_meters: np.ndarray = field(default_factory=lambda: np.array([[]], dtype=float))

    total_mission_time_seconds: Optional[float] = None
    total_energy_joules: Optional[float] = None
    aoi_per_area_seconds: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=float))
    is_feasible: Optional[bool] = None
    
    iteration_number: Optional[int] = 0

    # additional: slack variable λ_m (used for speed optimization)
    # has the same length as speeds_v_mps
    slacks_lambda_m: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=float))


    def __post_init__(self):
        # dimension check, etc.
        num_areas = len(self.visiting_sequence_pi)
        if num_areas > 0:
            expected_speed_len = num_areas + 1
            if self.speeds_v_mps.ndim == 0 or len(self.speeds_v_mps) != expected_speed_len:
                # when initialized, the length may be different, so check it at the actual usage time or just print a warning
                # print(f"Warning: speeds_v_mps length {len(self.speeds_v_mps)} does not match expected {expected_speed_len}")
                pass
            if self.slacks_lambda_m is not None and (self.slacks_lambda_m.ndim == 0 or len(self.slacks_lambda_m) != expected_speed_len):
                # print(f"Warning: slacks_lambda_m length {len(self.slacks_lambda_m)} does not match expected {expected_speed_len}")
                pass

            # the hover position array should be (total number of areas, 2)
            # self.hover_positions_q_tilde_meters.shape[0] is the actual number of areas
            # self.hover_positions_p_meters.shape[0]
            # this part depends on how the data is actually stored, so the verification logic is different
            # for example, if there is always information for all areas, compare it with num_total_areas
            pass


    def update_performance_metrics(self, time: float, energy: float, aois: np.ndarray, feasible: bool):
        self.total_mission_time_seconds = time
        self.total_energy_joules = energy
        self.aoi_per_area_seconds = aois.copy() if aois is not None else np.array([], dtype=float)
        self.is_feasible = feasible

    def get_num_visited_areas(self) -> int:
        """return the actual number of areas included in the visiting sequence"""
        return len(self.visiting_sequence_pi)

    def get_q_tilde_k_for_area_id(self, area_id: int) -> Optional[np.ndarray]:
        """return the data collection hovering position q̃_k for the given area_id"""
        if self.hover_positions_q_tilde_meters.ndim == 2 and \
           0 <= area_id < self.hover_positions_q_tilde_meters.shape[0]:
            return self.hover_positions_q_tilde_meters[area_id]
        return None

    def get_p_k_for_area_id(self, area_id: int) -> Optional[np.ndarray]:
        """return the data transmission hovering position p_k for the given area_id"""
        if self.hover_positions_p_meters.ndim == 2 and \
           0 <= area_id < self.hover_positions_p_meters.shape[0]:
            return self.hover_positions_p_meters[area_id]
        return None

    def get_speed_v0(self) -> Optional[float]:
        """return the flight speed v0 when no data is collected"""
        if self.speeds_v_mps.ndim > 0 and len(self.speeds_v_mps) > 0:
            return self.speeds_v_mps[0]
        return None

    def get_slack_lambda0(self) -> Optional[float]:
        """return the slack variable λ0 corresponding to v0"""
        if self.slacks_lambda_m is not None and self.slacks_lambda_m.ndim > 0 and len(self.slacks_lambda_m) > 0:
            return self.slacks_lambda_m[0]
        return None

    def get_speed_vk_for_sequence_index(self, sequence_index: int) -> Optional[float]:
        """
        return the speed v_k for the area at the given sequence index (starting from 0)
        (q̃_k -> p_k segment speed)
        corresponds to speeds_v_mps[sequence_index + 1]
        """
        if self.speeds_v_mps.ndim > 0 and 0 <= sequence_index < self.get_num_visited_areas() and \
           (sequence_index + 1) < len(self.speeds_v_mps):
            return self.speeds_v_mps[sequence_index + 1]
        return None

    def get_slack_lambdak_for_sequence_index(self, sequence_index: int) -> Optional[float]:
        """
        return the slack variable λ_k for the area at the given sequence index (starting from 0)
        corresponds to slacks_lambda_m[sequence_index + 1]
        """
        if self.slacks_lambda_m is not None and self.slacks_lambda_m.ndim > 0 and \
           0 <= sequence_index < self.get_num_visited_areas() and \
           (sequence_index + 1) < len(self.slacks_lambda_m):
            return self.slacks_lambda_m[sequence_index + 1]
        return None

    def get_path_segment_details(self, env: "Environment") -> List[Dict[str, Any]]:
        """
        return the details of the path segments
        each segment has the start position, end position, distance,
        applied speed index, applied slack index
        this method is used for energy calculation in SpeedOptimizationConstraintManager

        Args:
            env (Environment): environment information (initial UAV position, etc.)

        Returns:
            List[Dict[str, Any]]: a list of dictionaries, each containing the details of a path segment
                each dictionary has the keys 'start_pos', 'end_pos', 'distance',
                'speed_idx', 'slack_idx', 'segment_type'
                speed_idx/slack_idx are the indices of the self.speeds_v_mps/self.slacks_lambda_m array
        """
        segments = []
        if self.get_num_visited_areas() == 0:
            return segments

        current_pos = env.initial_uav_position_s0_meters

        # 1. initial position -> first q̃_k
        first_area_id = self.visiting_sequence_pi[0]
        q_tilde_first = self.get_q_tilde_k_for_area_id(first_area_id)
        if q_tilde_first is None: return [] # error handling

        dist = np.linalg.norm(q_tilde_first - current_pos)
        segments.append({
            'start_pos': current_pos.copy(), 'end_pos': q_tilde_first.copy(), 'distance': dist,
            'speed_idx': 0, 'slack_idx': 0, 'segment_type': 'initial_to_q'
        })
        current_pos = q_tilde_first

        for i in range(self.get_num_visited_areas()):
            area_id = self.visiting_sequence_pi[i]
            q_tilde_k = self.get_q_tilde_k_for_area_id(area_id)
            p_k = self.get_p_k_for_area_id(area_id)
            if q_tilde_k is None or p_k is None: return [] # error handling

            # 2. q̃_k -> p_k
            dist_q_to_p = np.linalg.norm(p_k - q_tilde_k)
            segments.append({
                'start_pos': q_tilde_k.copy(), 'end_pos': p_k.copy(), 'distance': dist_q_to_p,
                'speed_idx': i + 1, 'slack_idx': i + 1, 'segment_type': 'q_to_p'
            })
            current_pos = p_k

            # 3. p_k -> next q̃_{k+1}
            if i < self.get_num_visited_areas() - 1:
                next_area_id = self.visiting_sequence_pi[i+1]
                q_tilde_next = self.get_q_tilde_k_for_area_id(next_area_id)
                if q_tilde_next is None: return [] # error

                dist_p_to_next_q = np.linalg.norm(q_tilde_next - current_pos)
                segments.append({
                    'start_pos': current_pos.copy(), 'end_pos': q_tilde_next.copy(), 'distance': dist_p_to_next_q,
                    'speed_idx': 0, 'slack_idx': 0, 'segment_type': 'p_to_q'
                })
                current_pos = q_tilde_next

        # 4. last p_N -> initial position
        dist_last_p_to_initial = np.linalg.norm(env.initial_uav_position_s0_meters - current_pos)
        segments.append({
            'start_pos': current_pos.copy(), 'end_pos': env.initial_uav_position_s0_meters.copy(),
            'distance': dist_last_p_to_initial,
            'speed_idx': 0, 'slack_idx': 0, 'segment_type': 'p_to_initial'
        })
        return segments

    def copy_with_new_speeds_and_slacks(self, new_speeds: np.ndarray, new_slacks: Optional[np.ndarray] = None) -> "TrajectorySolution":
        """ return a new TrajectorySolution object with the current trajectory information and the updated speeds and slacks """
        return TrajectorySolution(
            visiting_sequence_pi=self.visiting_sequence_pi.copy(),
            speeds_v_mps=new_speeds.copy(),
            hover_positions_q_tilde_meters=self.hover_positions_q_tilde_meters.copy(),
            hover_positions_p_meters=self.hover_positions_p_meters.copy(),
            slacks_lambda_m=new_slacks.copy() if new_slacks is not None else None
            # performance metrics are not updated, so initialize them to None
        )