# workspace/optimization/visiting_seq_constraints.py
import numpy as np
from typing import List, Tuple, Optional

from model.uav import UAV
from model.environment import Environment
from model.trajectory import TrajectorySolution
from configs.data_classes import SimulationConfig

from utils import data_logger as logger

class VisitingSequenceConstraintManager:
    """
    Given a fixed visiting sequence and fixed UAV speed and hovering positions,
    check the constraints (AoI, energy) of P2.1 and calculate the objective function value (total mission completion time).
    """
    uav: UAV
    env: Environment
    sim_cfg: SimulationConfig
    fixed_trajectory_params: TrajectorySolution # include speed, q_all, p_all information

    def __init__(self,
                 uav_model: UAV,
                 env_model: Environment,
                 sim_config: SimulationConfig,
                 fixed_trajectory_params: TrajectorySolution): # receive TrajectorySolution
        self.uav = uav_model
        self.env = env_model
        self.sim_cfg = sim_config
        self.fixed_trajectory_params = fixed_trajectory_params

        if self.fixed_trajectory_params.speeds_v_mps is None or \
           self.fixed_trajectory_params.hover_positions_q_tilde_meters is None or \
           self.fixed_trajectory_params.hover_positions_p_meters is None:
            raise ValueError("fixed_trajectory_params must contain speeds, q_tilde_meters, and p_meters.")

    def _create_trajectory_for_sequence(self, visiting_sequence_pi: List[int]) -> TrajectorySolution:
        """
        Create a new TrajectorySolution object with the given visiting sequence and fixed speed/position stored in fixed_trajectory_params.
        """
        return TrajectorySolution(
            visiting_sequence_pi=visiting_sequence_pi,
            speeds_v_mps=self.fixed_trajectory_params.speeds_v_mps.copy(),
            hover_positions_q_tilde_meters=self.fixed_trajectory_params.hover_positions_q_tilde_meters.copy(),
            hover_positions_p_meters=self.fixed_trajectory_params.hover_positions_p_meters.copy(),
            slacks_lambda_m=None # usually not used for visiting sequence optimization
                                 # if needed, pass fixed_trajectory_params.slacks_lambda_m.copy()
        )

    def calculate_total_mission_time_for_sequence(self, visiting_sequence_pi: List[int]) -> float:
        """
        Calculate the total mission completion time for the given visiting sequence (P2.1's objective function).
        """
        if not visiting_sequence_pi:
            return float('inf')

        trajectory = self._create_trajectory_for_sequence(visiting_sequence_pi)
        total_time, _, _ = self.uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)
        return total_time

    def check_aoi_constraints_for_sequence(self, visiting_sequence_pi: List[int]) -> bool:
        """
        Check if all AoI constraints for all visited areas are satisfied for the given visiting sequence.
        """
        if not visiting_sequence_pi:
            return True

        trajectory = self._create_trajectory_for_sequence(visiting_sequence_pi)
        _, _, aois_per_visited_area = self.uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)

        aoi_limit = self.sim_cfg.general.aoi_limit_seconds
        
        if aois_per_visited_area is None or len(aois_per_visited_area) == 0:
            if not visiting_sequence_pi: return True
            return False

        for aoi_val in aois_per_visited_area:
            if aoi_val > aoi_limit + 1e-6 + self.sim_cfg.ga.aoi_tolerance:
                return False
        return True

    def check_energy_constraint_for_sequence(self, visiting_sequence_pi: List[int]) -> bool:
        """
        Check if the total energy consumption constraint is satisfied for the given visiting sequence.
        """
        if not visiting_sequence_pi:
            return True

        trajectory = self._create_trajectory_for_sequence(visiting_sequence_pi)
        _, total_energy, _ = self.uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)
        
        if total_energy is None:
            return False

        max_energy_limit = self.uav.config.energy.max_elimit_joule
        return total_energy <= max_energy_limit + 1e-6 + self.sim_cfg.ga.energy_tolerance

    def check_all_constraints_for_sequence(self, visiting_sequence_pi: List[int]) -> bool:
        """
        Check if all constraints (AoI, energy) are satisfied for the given visiting sequence.
        """
        if not self.check_aoi_constraints_for_sequence(visiting_sequence_pi):
            logger.debug(f"Sequence {visiting_sequence_pi} is infeasible due to AoI constraints.")
            return False
        if not self.check_energy_constraint_for_sequence(visiting_sequence_pi):
            logger.debug(f"Sequence {visiting_sequence_pi} is infeasible due to energy constraints.")
            return False
        return True