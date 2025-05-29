# optimization/random_sequence_optimization.py
import numpy as np
import random
from typing import List, Tuple, Optional

from model.uav import UAV
from model.environment import Environment
from model.trajectory import TrajectorySolution
from configs.data_classes import SimulationConfig
from utils import data_logger as logger
from .visiting_seq_constraints import VisitingSequenceConstraintManager

class RandomSequenceOptimizer:
    uav: UAV
    env: Environment
    sim_cfg: SimulationConfig
    fixed_trajectory_params: TrajectorySolution
    num_random_trials: int # how many random trials to perform

    constraint_manager: VisitingSequenceConstraintManager

    def __init__(self,
                 uav_model: UAV,
                 env_model: Environment,
                 sim_config: SimulationConfig,
                 fixed_trajectory_params: TrajectorySolution,
                 num_random_trials: int = 100): # example: 100 trials
        self.uav = uav_model
        self.env = env_model
        self.sim_cfg = sim_config
        self.fixed_trajectory_params = fixed_trajectory_params
        self.num_random_trials = num_random_trials
        self.constraint_manager = VisitingSequenceConstraintManager(
            uav_model, env_model, sim_config, fixed_trajectory_params
        )

    def _generate_random_permutation(self, num_items: int) -> List[int]:
        perm = list(range(num_items))
        random.shuffle(perm)
        return perm

    def optimize_sequence(self) -> Tuple[Optional[List[int]], Optional[float]]:
        logger.print_subheader("Starting Random Algorithm for Visiting Sequence Optimization")
        num_total_areas = self.env.get_total_monitoring_areas()
        if num_total_areas == 0:
            return [], 0.0

        best_random_sequence = None
        best_random_time = float('inf')

        for i in range(self.num_random_trials):
            random_sequence = self._generate_random_permutation(num_total_areas)
            
            # check constraints and calculate time
            if self.constraint_manager.check_all_constraints_for_sequence(random_sequence):
                current_time = self.constraint_manager.calculate_total_mission_time_for_sequence(random_sequence)
                if current_time < best_random_time:
                    best_random_time = current_time
                    best_random_sequence = random_sequence
            
            if (i + 1) % (self.num_random_trials // 10 or 1) == 0:
                 logger.debug(f"Random Trial {i+1}/{self.num_random_trials}. Current best time: {best_random_time:.4f}")


        if best_random_sequence is None:
            logger.warning("Random algorithm could not find any feasible solution.")
            return None, None
            
        logger.info(f"Random algorithm finished. Best sequence: {best_random_sequence}, Min time: {best_random_time:.4f}")
        return best_random_sequence, best_random_time