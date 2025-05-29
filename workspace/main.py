# workspace/main.py

import numpy as np
import random
import argparse

from model.environment import Environment
from model.uav import UAV
from model.trajectory import TrajectorySolution
from optimization.speed_optimization import SpeedOptimizerSCA
from optimization.visiting_seq_optimization import VisitingSequenceOptimizerGA
from optimization.hovering_pos_optimization import HoveringPositionOptimizerSCA
from optimization.greedy_sequence_optimization import GreedySequenceOptimizer
from optimization.random_sequence_optimization import RandomSequenceOptimizer

from typing import List, Optional 

from utils import data_logger as logger
from configs.config_loader import (
    get_experiment_fixed_path_params,
    get_uav_config,
    get_environment_config,
    get_simulation_config
)

from utils.plotting_results import plot_trajectory
from utils.plot_time_curve import plot_time_curve
from utils.plot_speed_areas import plot_speed_vs_area_index
from utils.plot_aoi_areas import plot_aoi_per_area

def log_trajectory_result(trajectory, uav, algorithm_name="Algorithm"):
    """Log trajectory results"""
    logger.info(f"--- {algorithm_name} Result ---")
    logger.info(f"Visiting Sequence: {trajectory.visiting_sequence_pi}")
    logger.info(f"Speeds (m/s): {trajectory.speeds_v_mps}")
    final_time, final_energy, final_aois = uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)
    logger.info(f"Final mission time: {final_time:.4f}s, Energy: {final_energy:.4f}J")
    if final_aois is not None and len(final_aois) > 0:
        logger.info(f"AOIs per area: {final_aois}")
    else:
        logger.info("AOIs per area: Not available or no areas visited.")
    logger.info(f"--- End of {algorithm_name} Result ---")


def run_experiment(algorithm_to_run: str,
                   energy_limit_kj: float = None,
                   aoi_limit_s: Optional[float] = None,
                   plot: bool = True
                  ):
    # --- Common initialization ---
    uav_cfg = get_uav_config()
    env_cfg = get_environment_config()
    sim_cfg = get_simulation_config()
    exp_params = get_experiment_fixed_path_params() # for initial/fixed path
    logger.info(f"All configurations loaded for algorithm: {algorithm_to_run}")

    if energy_limit_kj is not None:
        uav_cfg.energy.max_elimit_joule = energy_limit_kj * 1000.0
        logger.info(f"Applied energy limit: {uav_cfg.energy.max_elimit_joule} J ({energy_limit_kj} KJ)")
    
    if aoi_limit_s is not None:
        if hasattr(sim_cfg, 'general') and hasattr(sim_cfg.general, 'aoi_limit_seconds'):
            sim_cfg.general.aoi_limit_seconds = aoi_limit_s
            logger.info(f"Applied AoI limit from argument: {aoi_limit_s}s (overwriting simulation.yaml)")
        else:
            logger.warning(f"Could not apply AoI limit {aoi_limit_s}s. "
                           f"sim_cfg.general.aoi_limit_seconds not found. "
                           f"Ensure simulation.yaml structure is correct and loaded properly.")
    else:
        if hasattr(sim_cfg, 'general') and hasattr(sim_cfg.general, 'aoi_limit_seconds'):
            logger.info(f"Using default AoI limit from simulation.yaml: {sim_cfg.general.aoi_limit_seconds} s")
        else:
            logger.warning("No AoI limit provided and default not found in simulation.yaml.")

    
    if sim_cfg.general.random_seed is not None:
        np.random.seed(sim_cfg.general.random_seed)
        random.seed(sim_cfg.general.random_seed)
        logger.info(f"Global random seed set to: {sim_cfg.general.random_seed}")

    env = Environment(config=env_cfg, sim_config=sim_cfg)
    uav = UAV(uav_cfg, env)


    if algorithm_to_run == "AoI-EaTO":
        logger.print_header(f"UAV Trajectory Optimization Simulation via {algorithm_to_run}")
        trajectory = TrajectorySolution(
            visiting_sequence_pi=exp_params.fixed_visiting_sequence,
            speeds_v_mps=np.array(exp_params.initial_speeds_mps, dtype=float),
            hover_positions_q_tilde_meters=np.array(exp_params.fixed_hover_q_tilde_meters, dtype=float),
            hover_positions_p_meters=np.array(exp_params.fixed_hover_p_meters, dtype=float),
            slacks_lambda_m=np.array(exp_params.initial_slacks_lambda, dtype=float),
            iteration_number=0
        )

        # Outer loop: Algorithm 4 (AoI-EaTO)
        max_outer = sim_cfg.aoi_eato.max_iterations
        tol = sim_cfg.aoi_eato.convergence_threshold_seconds
        prev_obj = float('inf')

        logger.print_subheader("Initial trajectory result (for AoI-EaTO)")
        log_trajectory_result(trajectory, uav, "Initial AoI-EaTO")
        if plot:
            plot_trajectory(trajectory, env, uav, show=True, save_path=f"./results/trajectory_aoi_eato_iter_0.png")

        min_time = float('inf')
        min_time_trajectory = trajectory
        times_list = []

        for l in range(max_outer):
            trajectory.iteration_number = l
            logger.print_subheader(f"AoI-EaTO Iteration {l+1}/{max_outer}")

            # Stage 1: Speed optimization (Algorithm 1)
            logger.info("Starting Stage 1: Speed Optimization (SCA)")
            speed_opt = SpeedOptimizerSCA(
                uav_model=uav,
                env_model=env,
                sim_config=sim_cfg,
                trajectory_solution=trajectory,
                sca_convergence_tol=tol / 10.0
            )
            v_opt, slacks_opt, _ = speed_opt.optimize_speeds()
            if v_opt is None:
                logger.error("Speed optimization failed. Using previous speeds.")
            else:
                logger.info("Speed optimization completed successfully.")
                trajectory.speeds_v_mps = v_opt
                trajectory.slacks_lambda_m = slacks_opt if slacks_opt is not None else trajectory.slacks_lambda_m

            # Stage 2a: Visiting sequence via GA (Algorithm 2)
            logger.info("Starting Stage 2a: Visiting Sequence Optimization (GA)")
            seq_optimizer_ga = VisitingSequenceOptimizerGA(
                uav_model=uav,
                env_model=env,
                sim_config=sim_cfg,
                fixed_trajectory_params=trajectory
            )
            seq_opt, _ = seq_optimizer_ga.optimize_sequence()
            if seq_opt is None:
                logger.error("Visiting sequence optimization failed. Using previous sequence.")
            else:
                logger.info("Visiting sequence optimization completed successfully.")
                trajectory.visiting_sequence_pi = seq_opt
            
            # Stage 2b: Hovering positions via SCA (Algorithm 3)
            logger.info("Starting Stage 2b: Hovering Position Optimization (SCA)")
            hover_opt = HoveringPositionOptimizerSCA(
                uav_model=uav,
                env_model=env,
                sim_config=sim_cfg,
                fixed_trajectory_params=trajectory
            )
            q_opt, p_opt, _ = hover_opt.optimize_hovering_positions()
            if q_opt is None or p_opt is None:
                logger.error("Hovering position optimization failed. Using previous positions.")
            else:
                logger.info("Hovering position optimization completed successfully.")
                trajectory.hover_positions_q_tilde_meters = q_opt
                trajectory.hover_positions_p_meters = p_opt


            current_time, _, _ = uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)
            logger.info(f"AoI-EaTO Iteration {l+1} mission time: {current_time:.4f}s")

            if abs(prev_obj - current_time) < tol and l > 0:
                logger.info("AoI-EaTO algorithm converged.")
                # break
            prev_obj = current_time

            logger.print_subheader(f"AoI-EaTO Iteration {l+1} trajectory result")
            log_trajectory_result(trajectory, uav, f"AoI-EaTO Iter {l+1}")
            if plot:
                plot_trajectory(trajectory, env, uav, show=True, save_path=f"./results/trajectory_aoi_eato_iter_{l+1}.png")
            
            if current_time < min_time:
                min_time = current_time
                min_time_trajectory = trajectory
            times_list.append(min_time)
        logger.print_subheader("Final Optimized Trajectory Result (AoI-EaTO)")
        log_trajectory_result(trajectory, uav, "Final AoI-EaTO")
        if plot:
            plot_time_curve(times_list, save_path=f"./results/time_curve_aoi_eato.png")
            plot_speed_vs_area_index(min_time_trajectory, env, save_path=f"./results/speed_vs_area_index_aoi_eato_{int(uav.get_max_energy_kjoules())}kj.png")
            aoi_values = uav.calculate_total_mission_time_energy_aois_for_trajectory(min_time_trajectory)[2]
            plot_aoi_per_area(aoi_values, save_path=f"./results/aoi_per_area_aoi_eato_{int(uav.get_max_energy_kjoules())}kj.png")

        return min_time_trajectory, uav, env

    elif algorithm_to_run == "Random":
        logger.print_header(f"UAV Trajectory Optimization Simulation via {algorithm_to_run}")

        # Initial UAV path W0
        # Hovering positions are fixed, speeds are initialized and optimized using SCA
        initial_trajectory_for_random = TrajectorySolution(
            visiting_sequence_pi=[], # Random algorithm determines the sequence
            speeds_v_mps=np.array(exp_params.initial_speeds_mps, dtype=float),
            hover_positions_q_tilde_meters=np.array(exp_params.fixed_hover_q_tilde_meters, dtype=float),
            hover_positions_p_meters=np.array(exp_params.fixed_hover_p_meters, dtype=float),
            slacks_lambda_m=np.array(exp_params.initial_slacks_lambda, dtype=float)
        )

        # 1. Random visiting sequence determination
        logger.info("Starting Stage 1: Visiting Sequence Optimization (Random)")
        num_trials_random = sim_cfg.ga.population_size * sim_cfg.ga.num_generations # Try GA's population_size times (adjustable)

        random_opt = RandomSequenceOptimizer(
            uav_model=uav,
            env_model=env,
            sim_config=sim_cfg,
            fixed_trajectory_params=initial_trajectory_for_random, # Pass fixed speed/position information
            num_random_trials=num_trials_random
        )
        opt_sequence_random, _ = random_opt.optimize_sequence() # Time value is for internal logging

        if opt_sequence_random is None:
            logger.error("Random algorithm failed to find a feasible sequence")
            opt_sequence_random = []
            return None, None, None
            
        # Update trajectory with determined visiting sequence (speeds are still initial values)
        trajectory_after_random_seq = TrajectorySolution(
            visiting_sequence_pi=opt_sequence_random,
            speeds_v_mps=initial_trajectory_for_random.speeds_v_mps.copy(),
            hover_positions_q_tilde_meters=initial_trajectory_for_random.hover_positions_q_tilde_meters.copy(),
            hover_positions_p_meters=initial_trajectory_for_random.hover_positions_p_meters.copy(),
            slacks_lambda_m=initial_trajectory_for_random.slacks_lambda_m.copy()
        )
        logger.info(f"Random - Sequence found: {opt_sequence_random}")

        # 2. Speed optimization (SCA)
        logger.info("Starting Stage 2: Speed Optimization (SCA) for Random Sequence")
        speed_opt_random = SpeedOptimizerSCA(
            uav_model=uav,
            env_model=env,
            sim_config=sim_cfg,
            trajectory_solution=trajectory_after_random_seq, # Determined sequence and fixed hovering positions, initial speeds
            sca_convergence_tol=sim_cfg.aoi_eato.convergence_threshold_seconds / 10.0 # Same convergence condition as AoI-EaTO
        )
        v_opt_random, slacks_opt_random, _ = speed_opt_random.optimize_speeds()

        final_trajectory_random: TrajectorySolution
        if v_opt_random is None:
            logger.error("Random - Speed optimization failed. Using initial speeds for the found sequence.")
            final_trajectory_random = trajectory_after_random_seq
        else:
            logger.info("Random - Speed optimization completed successfully.")
            final_trajectory_random = TrajectorySolution(
                visiting_sequence_pi=opt_sequence_random,
                speeds_v_mps=v_opt_random,
                hover_positions_q_tilde_meters=initial_trajectory_for_random.hover_positions_q_tilde_meters.copy(),
                hover_positions_p_meters=initial_trajectory_for_random.hover_positions_p_meters.copy(),
                slacks_lambda_m=slacks_opt_random if slacks_opt_random is not None else initial_trajectory_for_random.slacks_lambda_m.copy()
            )

        logger.print_subheader(f"Final Optimized Trajectory Result ({algorithm_to_run})")
        log_trajectory_result(final_trajectory_random, uav, algorithm_to_run)
        if plot:
            plot_trajectory(final_trajectory_random, env, uav, show=True, save_path=f"./results/trajectory_{algorithm_to_run.lower()}.png")
            plot_speed_vs_area_index(final_trajectory_random, env, save_path=f"./results/speed_vs_area_index_{algorithm_to_run.lower()}_{int(uav.get_max_energy_kjoules())}kj.png")
            aoi_values = uav.calculate_total_mission_time_energy_aois_for_trajectory(final_trajectory_random)[2]
            plot_aoi_per_area(aoi_values, save_path=f"./results/aoi_per_area_{algorithm_to_run.lower()}_{int(uav.get_max_energy_kjoules())}kj.png")

        return final_trajectory_random, uav, env
    
    elif algorithm_to_run == "Greedy":
        logger.print_header(f"UAV Trajectory Optimization Simulation via {algorithm_to_run}")
        # Initial trajectory setup for Greedy algorithm
        initial_trajectory_for_greedy = TrajectorySolution(
            visiting_sequence_pi=[], # Greedy algorithm determines the sequence
            speeds_v_mps=np.array(exp_params.initial_speeds_mps, dtype=float),
            hover_positions_q_tilde_meters=np.array(exp_params.fixed_hover_q_tilde_meters, dtype=float),
            hover_positions_p_meters=np.array(exp_params.fixed_hover_p_meters, dtype=float),
            slacks_lambda_m=np.array(exp_params.initial_slacks_lambda, dtype=float)
        )

        # 1. Greedy visiting sequence determination
        logger.info("Starting Stage 1: Visiting Sequence Optimization (Greedy)")
        greedy_opt = GreedySequenceOptimizer(
            uav_model=uav,
            env_model=env,
            sim_config=sim_cfg,
            fixed_trajectory_params=initial_trajectory_for_greedy
        )
        opt_sequence_greedy, _ = greedy_opt.optimize_sequence()

        if opt_sequence_greedy is None:
            logger.error("Greedy algorithm failed to find a feasible sequence. Exiting.")
            return None, None, None

        trajectory_after_greedy_seq = TrajectorySolution(
            visiting_sequence_pi=opt_sequence_greedy,
            speeds_v_mps=initial_trajectory_for_greedy.speeds_v_mps.copy(),
            hover_positions_q_tilde_meters=initial_trajectory_for_greedy.hover_positions_q_tilde_meters.copy(),
            hover_positions_p_meters=initial_trajectory_for_greedy.hover_positions_p_meters.copy(),
            slacks_lambda_m=initial_trajectory_for_greedy.slacks_lambda_m.copy()
        )
        logger.info(f"Greedy - Sequence found: {opt_sequence_greedy}")

        # 2. Speed optimization (SCA)
        logger.info("Starting Stage 2: Speed Optimization (SCA) for Greedy Sequence")
        speed_opt_greedy = SpeedOptimizerSCA(
            uav_model=uav,
            env_model=env,
            sim_config=sim_cfg,
            trajectory_solution=trajectory_after_greedy_seq,
            sca_convergence_tol=sim_cfg.aoi_eato.convergence_threshold_seconds / 10.0
        )
        v_opt_greedy, slacks_opt_greedy, _ = speed_opt_greedy.optimize_speeds()

        final_trajectory_greedy: TrajectorySolution
        if v_opt_greedy is None:
            logger.error("Greedy - Speed optimization failed. Using initial speeds for the found sequence.")
            final_trajectory_greedy = trajectory_after_greedy_seq
        else:
            logger.info("Greedy - Speed optimization completed successfully.")
            final_trajectory_greedy = TrajectorySolution(
                visiting_sequence_pi=opt_sequence_greedy,
                speeds_v_mps=v_opt_greedy,
                hover_positions_q_tilde_meters=initial_trajectory_for_greedy.hover_positions_q_tilde_meters.copy(),
                hover_positions_p_meters=initial_trajectory_for_greedy.hover_positions_p_meters.copy(),
                slacks_lambda_m=slacks_opt_greedy if slacks_opt_greedy is not None else initial_trajectory_for_greedy.slacks_lambda_m.copy()
            )
        
        logger.print_subheader(f"Final Optimized Trajectory Result ({algorithm_to_run})")
        log_trajectory_result(final_trajectory_greedy, uav, algorithm_to_run)
        if plot:
            plot_trajectory(final_trajectory_greedy, env, uav, show=True, save_path=f"./results/trajectory_{algorithm_to_run.lower()}.png")
            plot_speed_vs_area_index(final_trajectory_greedy, env, save_path=f"./results/speed_vs_area_index_{algorithm_to_run.lower()}_{int(uav.get_max_energy_kjoules())}kj.png")
            aoi_values = uav.calculate_total_mission_time_energy_aois_for_trajectory(final_trajectory_greedy)[2]
            plot_aoi_per_area(aoi_values, save_path=f"./results/aoi_per_area_{algorithm_to_run.lower()}_{int(uav.get_max_energy_kjoules())}kj.png")
        return final_trajectory_greedy, uav, env

    else:
        logger.error(f"Unsupported algorithm specified: {algorithm_to_run}")
        return None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Trajectory Optimization Simulation")
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="AoI-EaTO",
        choices=["AoI-EaTO", "Random", "Greedy"],
        help="Algorithm to run for trajectory optimization (default: AoI-EaTO)"
    )
    
    args = parser.parse_args()
    
    # Run experiment with selected algorithm
    run_experiment(args.algorithm)