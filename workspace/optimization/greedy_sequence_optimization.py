# workspace/optimization/greedy_sequence_optimization.py
import numpy as np
from typing import List, Tuple, Optional

from model.uav import UAV
from model.environment import Environment
from model.trajectory import TrajectorySolution
from configs.data_classes import SimulationConfig
from utils import data_logger as logger
from .visiting_seq_constraints import VisitingSequenceConstraintManager

class GreedySequenceOptimizer:
    """
    Greedy Algorithm for Visiting Sequence Optimization.
    When selecting the next area to visit from the current position,
    the Greedy algorithm selects the most "greedy" option based on a specific criterion (e.g., the closest distance).
    The determined sequence can then be optimized externally with speed optimization.
    This class determines the sequence only assuming fixed speed and hovering positions.
    """
    uav: UAV
    env: Environment
    sim_cfg: SimulationConfig
    # In Greedy, the speed and hovering position are fixed, so only the sequence is determined.
    # This information is passed through fixed_trajectory_params.
    fixed_trajectory_params: TrajectorySolution

    constraint_manager: VisitingSequenceConstraintManager

    def __init__(self,
                 uav_model: UAV,
                 env_model: Environment,
                 sim_config: SimulationConfig,
                 fixed_trajectory_params: TrajectorySolution):
        self.uav = uav_model
        self.env = env_model
        self.sim_cfg = sim_config
        self.fixed_trajectory_params = fixed_trajectory_params

        if self.fixed_trajectory_params.speeds_v_mps is None or \
           self.fixed_trajectory_params.hover_positions_q_tilde_meters is None or \
           self.fixed_trajectory_params.hover_positions_p_meters is None:
            raise ValueError("fixed_trajectory_params for GreedySequenceOptimizer must contain "
                             "speeds, q_tilde_meters, and p_meters for cost/constraint evaluation.")

        # Reuse ConstraintManager for constraint checking and time calculation
        self.constraint_manager = VisitingSequenceConstraintManager(
            uav_model, env_model, sim_config, fixed_trajectory_params
        )

    def _calculate_cost_to_next_area(self,
                                     current_p_k_pos_meters: np.ndarray, # Current
                                     next_area_id_to_visit: int
                                    ) -> float:
        """
        Calculate the cost (here, Euclidean distance) from the current UAV position to the data collection position (q_tilde) of the next area to visit.
        A more sophisticated cost function can also be used (e.g., flight time, energy consumption, etc.).

        Args:
            current_p_k_pos_meters (np.ndarray): Current UAV position (previous area's p_k or initial s0).
            next_area_id_to_visit (int): ID of the next candidate area to visit.

        Returns:
            float: Cost value (smaller is better).
        """
        q_tilde_next = self.fixed_trajectory_params.get_q_tilde_k_for_area_id(next_area_id_to_visit)
        if q_tilde_next is None:
            logger.warning(f"Greedy cost calc: q_tilde not found for area {next_area_id_to_visit}")
            return float('inf') # No information for this area

        distance = np.linalg.norm(q_tilde_next - current_p_k_pos_meters)
        return distance

    def optimize_sequence(self) -> Tuple[Optional[List[int]], Optional[float]]:
        """
        Optimize the visiting sequence using the Greedy algorithm.

        Returns:
            Tuple[Optional[List[int]], Optional[float]]:
                - Optimized visiting sequence list (area_id based, 0-indexed).
                - Total mission completion time for the corresponding sequence (fixed speed/position based).
                - Failure (None, None).
        """
        logger.print_subheader("Starting Greedy Algorithm for Visiting Sequence Optimization")
        num_total_areas = self.env.get_total_monitoring_areas()
        if num_total_areas == 0:
            logger.info("Greedy: No monitoring areas to visit.")
            return [], 0.0

        unvisited_area_ids = list(range(num_total_areas)) # 0 to N-1 area IDs
        greedy_sequence: List[int] = []

        # Initial UAV position is s0
        current_uav_logical_pos = self.env.initial_uav_position_s0_meters.copy()

        for step in range(num_total_areas):
            if not unvisited_area_ids:
                logger.warning("Greedy: Ran out of unvisited areas unexpectedly.")
                break # No more areas to visit

            best_next_area_id = -1
            min_cost_to_next = float('inf')

            # Select the next area to visit
            for candidate_area_id in unvisited_area_ids:
                cost = self._calculate_cost_to_next_area(current_uav_logical_pos, candidate_area_id)

                if cost < min_cost_to_next:
                    # Checking if this selection satisfies all constraints is very complex.
                    # Greedy usually only considers local optimality.
                    # Here, we just select based on cost, and check constraints for the final sequence.
                    min_cost_to_next = cost
                    best_next_area_id = candidate_area_id
            
            if best_next_area_id != -1:
                greedy_sequence.append(best_next_area_id)
                unvisited_area_ids.remove(best_next_area_id)
                
                # Update the current UAV position for the next search
                # Assume the UAV has moved to the data transmission position (p_k) of the current selected area
                p_k_of_current_best = self.fixed_trajectory_params.get_p_k_for_area_id(best_next_area_id)
                if p_k_of_current_best is None:
                    logger.error(f"Greedy: p_k not found for selected area {best_next_area_id}. Cannot update current_uav_pos.")
                    return None, None # Serious error
                current_uav_logical_pos = p_k_of_current_best.copy()
                logger.debug(f"Greedy step {step+1}: Selected area {best_next_area_id}, cost {min_cost_to_next:.2f}. Next logical pos: {current_uav_logical_pos}")

            else:
                # This can happen when the cost is inf for all remaining areas (e.g., no q_tilde information)
                logger.warning(f"Greedy: Could not find a next area to visit at step {step+1}. "
                               f"Unvisited: {unvisited_area_ids}. Sequence might be incomplete.")
                break # No more progress

        if len(greedy_sequence) != num_total_areas:
            logger.warning(f"Greedy algorithm could not visit all areas. Visited: {greedy_sequence}")
            # Need to decide whether to return a partial solution or treat it as failure
            # Here, we just evaluate the generated partial sequence
            if not greedy_sequence : return None, None


        # Evaluate the time and constraints for the final generated sequence
        # (using the fixed speed and hovering positions stored in fixed_trajectory_params)
        final_time = self.constraint_manager.calculate_total_mission_time_for_sequence(greedy_sequence)
        
        if final_time == float('inf') or not self.constraint_manager.check_all_constraints_for_sequence(greedy_sequence):
            logger.warning(f"Greedy sequence {greedy_sequence} is infeasible based on final check. Time: {final_time}")
            # It might not satisfy all constraints because we just followed the closest one
            return None, None
            
        logger.info(f"Greedy algorithm finished. Sequence: {greedy_sequence}, Min time (with fixed params): {final_time:.4f}")
        return greedy_sequence, final_time