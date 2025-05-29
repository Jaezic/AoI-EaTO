# workspace/optimization/visiting_seq_optimization.py
import numpy as np
import random
from typing import List, Tuple, Optional

from model.uav import UAV
from model.environment import Environment
from model.trajectory import TrajectorySolution # 추가
from configs.data_classes import SimulationConfig
from .visiting_seq_constraints import VisitingSequenceConstraintManager
from utils import data_logger as logger

class VisitingSequenceOptimizerGA:
    """
    Algorithm 2: GA-Based Visiting Sequence Optimization for P2.1.
    Given a fixed UAV speed and hovering positions, find the optimal visiting sequence to minimize the total mission completion time.
    """
    uav: UAV
    env: Environment
    sim_cfg: SimulationConfig
    
    fixed_trajectory_params: TrajectorySolution # include speed, q_all, p_all information

    population_size: int
    num_generations: int
    mutation_rate: float
    crossover_rate: float
    elitism_count: int

    constraint_manager: VisitingSequenceConstraintManager
    
    population: List[List[int]]
    fitness_scores: List[float]

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

        ga_params = sim_config.ga
        self.population_size = ga_params.population_size
        self.num_generations = ga_params.num_generations
        self.mutation_rate = ga_params.mutation_rate
        self.crossover_rate = ga_params.crossover_rate
        self.elitism_count = int(self.population_size * ga_params.selection_pressure_ratio)
        if self.elitism_count == 0 and self.population_size > 0:
             self.elitism_count = 1 

        self.constraint_manager = VisitingSequenceConstraintManager(
            uav_model, env_model, sim_config,
            self.fixed_trajectory_params # pass TrajectorySolution object
        )
        self.population = []
        self.fitness_scores = []

    def _generate_random_permutation(self, num_items: int) -> List[int]:
        perm = list(range(num_items))
        random.shuffle(perm)
        return perm

    def _initialize_population(self):
        self.population = []
        num_total_areas = self.env.get_total_monitoring_areas()
        
        attempts = 0
        max_attempts_per_chromosome = 200
        total_max_attempts = self.population_size * max_attempts_per_chromosome

        for _ in range(self.population_size):
            chromosome = None
            for _ in range(max_attempts_per_chromosome):
                attempts +=1
                temp_chromosome = self._generate_random_permutation(num_total_areas)
                if self.constraint_manager.check_all_constraints_for_sequence(temp_chromosome):
                    chromosome = temp_chromosome
                    break
                if attempts > total_max_attempts and len(self.population) < self.population_size / 2:
                    logger.warning("GA Init: Taking too long to find feasible solutions. Relaxing for remaining.")
                    chromosome = temp_chromosome
                    break
            
            if chromosome is None:
                chromosome = self._generate_random_permutation(num_total_areas)

            self.population.append(chromosome)
            
        if len(self.population) < self.population_size:
            logger.warning(f"GA Init: Population size {len(self.population)}, desired {self.population_size}. "
                           f"Not all initial solutions might be feasible.")

    def _calculate_fitness(self, chromosome: List[int]) -> float:
        if not self.constraint_manager.check_all_constraints_for_sequence(chromosome):
            return float('inf')
        return self.constraint_manager.calculate_total_mission_time_for_sequence(chromosome)

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        size = len(parent1)
        child1, child2 = [-1]*size, [-1]*size
        start, end = sorted(random.sample(range(size), 2))

        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]

        p2_idx = 0
        for i in range(size):
            if child1[i] == -1:
                while parent2[p2_idx] in child1:
                    p2_idx += 1
                child1[i] = parent2[p2_idx]
                p2_idx += 1
        p1_idx = 0
        for i in range(size):
            if child2[i] == -1:
                while parent1[p1_idx] in child2:
                    p1_idx += 1
                child2[i] = parent1[p1_idx]
                p1_idx += 1
        return child1, child2

    def _swap_mutation(self, chromosome: List[int]) -> List[int]:
        mutated_chromosome = list(chromosome)
        idx1, idx2 = random.sample(range(len(mutated_chromosome)), 2)
        mutated_chromosome[idx1], mutated_chromosome[idx2] = mutated_chromosome[idx2], mutated_chromosome[idx1]
        return mutated_chromosome

    def optimize_sequence(self) -> Tuple[Optional[List[int]], Optional[float]]:
        logger.print_subheader("Starting GA for Visiting Sequence Optimization")
        num_total_areas = self.env.get_total_monitoring_areas()
        if num_total_areas == 0:
            logger.info("GA: No monitoring areas to visit.")
            return [], 0.0

        # The visiting sequence in fixed_trajectory_params is not used for GA optimization.
        # GA generates a new visiting sequence.
        # fixed_trajectory_params provides fixed speed and hovering positions.
        
        self._initialize_population()
        if not self.population:
            logger.error("GA: Failed to initialize population.")
            return None, None

        best_chromosome_overall = None
        best_fitness_overall = float('inf')

        for generation in range(self.num_generations):
            self.fitness_scores = [self._calculate_fitness(chromo) for chromo in self.population]
            current_best_idx = np.argmin(self.fitness_scores)
            if self.fitness_scores[current_best_idx] < best_fitness_overall:
                best_fitness_overall = self.fitness_scores[current_best_idx]
                best_chromosome_overall = self.population[current_best_idx][:]
            
            new_population = []
            
            sorted_indices = np.argsort(self.fitness_scores)
            for i in range(self.elitism_count):
                if i < len(sorted_indices):
                     new_population.append(self.population[sorted_indices[i]][:])

            while len(new_population) < self.population_size:
                parent1 = random.choice(self.population)
                parent2 = random.choice(self.population)
                
                child1, child2 = parent1[:], parent2[:]

                if random.random() < self.crossover_rate:
                    child1_cand, child2_cand = self._order_crossover(parent1, parent2)
                    if self.constraint_manager.check_all_constraints_for_sequence(child1_cand):
                        child1 = child1_cand
                    if self.constraint_manager.check_all_constraints_for_sequence(child2_cand):
                        child2 = child2_cand
                
                if random.random() < self.mutation_rate:
                    child1_mut_cand = self._swap_mutation(child1)
                    if self.constraint_manager.check_all_constraints_for_sequence(child1_mut_cand):
                        child1 = child1_mut_cand
                
                if random.random() < self.mutation_rate:
                    child2_mut_cand = self._swap_mutation(child2)
                    if self.constraint_manager.check_all_constraints_for_sequence(child2_mut_cand):
                        child2 = child2_mut_cand

                if len(new_population) < self.population_size:
                    new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population

            if (generation + 1) % (self.num_generations // 10 or 1) == 0 :
                logger.info(f"[Sequence] GA Gen {generation+1}/{self.num_generations}. Current Best Fitness: {best_fitness_overall:.4f}")
        
        self.fitness_scores = [self._calculate_fitness(chromo) for chromo in self.population]
        final_best_idx = np.argmin(self.fitness_scores)
        if self.fitness_scores[final_best_idx] < best_fitness_overall:
            best_fitness_overall = self.fitness_scores[final_best_idx]
            best_chromosome_overall = self.population[final_best_idx][:]

        logger.info(f"GA finished. Best sequence: {best_chromosome_overall}, Min time: {best_fitness_overall:.4f}")
        if best_fitness_overall == float('inf'):
            logger.warning("GA could not find any feasible solution for the visiting sequence.")
            return None, None
            
        return best_chromosome_overall, best_fitness_overall