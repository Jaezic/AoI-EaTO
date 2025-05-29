# workspace/main.py

import numpy as np
import random

from model.environment import Environment
from model.uav import UAV
from model.trajectory import TrajectorySolution
from optimization.speed_optimization import SpeedOptimizerSCA
from utils import data_logger as logger
from configs.config_loader import get_experiment_fixed_path_params, get_uav_config, get_environment_config, get_simulation_config
from optimization.visiting_seq_optimization import VisitingSequenceOptimizerGA
from optimization.hovering_pos_optimization import HoveringPositionOptimizerSCA

def run_experiment():
    logger.print_header("UAV Trajectory Optimization Simulation")
    uav_cfg = get_uav_config()
    env_cfg = get_environment_config()
    sim_cfg = get_simulation_config()
    exp_params = get_experiment_fixed_path_params()
    logger.info("Configurations loaded successfully.")

    # (선택적) 랜덤 시드 설정
    if sim_cfg.general.random_seed is not None:
        np.random.seed(sim_cfg.general.random_seed)
        random.seed(sim_cfg.general.random_seed)
        logger.info(f"Random seed set to: {sim_cfg.general.random_seed}")

    logger.info("Initializing experiment components...")
    env = Environment(env_cfg)
    uav = UAV(uav_cfg, env)

    fixed_path_for_p1 = TrajectorySolution(
        visiting_sequence_pi=exp_params.fixed_visiting_sequence,
        speeds_v_mps=np.array(exp_params.initial_speeds_mps, dtype=float),
        hover_positions_q_tilde_meters=np.array(exp_params.fixed_hover_q_tilde_meters, dtype=float),
        hover_positions_p_meters=np.array(exp_params.fixed_hover_p_meters, dtype=float),
        slacks_lambda_m=np.array(exp_params.initial_slacks_lambda, dtype=float)
    )

    logger.info("Loaded fixed path parameters from experiment_params.yaml for P1.2 speed optimization.")
    logger.debug(f"Fixed path sequence: {fixed_path_for_p1.visiting_sequence_pi}")
    logger.debug(f"Initial speeds for P1.2: {fixed_path_for_p1.speeds_v_mps}")

    speed_optimizer = SpeedOptimizerSCA(
        uav_model=uav,
        env_model=env,
        sim_config=sim_cfg,
        trajectory_solution=fixed_path_for_p1,
        max_sca_iterations=sim_cfg.aoi_eato.max_iterations,
        sca_convergence_tol=sim_cfg.aoi_eato.convergence_threshold_seconds / 10.0
    )

    logger.print_subheader("Running P1.2 Speed Optimization")
    opt_speeds, opt_slacks, opt_time = speed_optimizer.optimize_speeds()

    print(f"opt_speeds: {opt_speeds}")
    print(f"opt_slacks: {opt_slacks}")
    print(f"opt_time: {opt_time}")

    fixed_path_for_p1.speeds_v_mps = opt_speeds
    fixed_path_for_p1.slacks_lambda_m = opt_slacks

    if opt_speeds is None:
        logger.error("Failed to optimize speeds. Exiting.")
        return
    
    trajectory_solution_p1 = TrajectorySolution(
        visiting_sequence_pi=fixed_path_for_p1.visiting_sequence_pi,
        speeds_v_mps=opt_speeds,
        hover_positions_q_tilde_meters=fixed_path_for_p1.hover_positions_q_tilde_meters,
        hover_positions_p_meters=fixed_path_for_p1.hover_positions_p_meters,
        slacks_lambda_m=opt_slacks
    )

    logger.info("Parameters for P2.1 Visiting Sequence Optimization (GA):")
    logger.debug(f"  Fixed speeds: {trajectory_solution_p1.speeds_v_mps}")
    logger.debug(f"  Fixed q_tilde_all (shape): {trajectory_solution_p1.hover_positions_q_tilde_meters.shape}")
    logger.debug(f"  Fixed p_all (shape): {trajectory_solution_p1.hover_positions_p_meters.shape}")

    sequence_optimizer = VisitingSequenceOptimizerGA(
        uav_model=uav,
        env_model=env,
        sim_config=sim_cfg,
        fixed_trajectory_params=trajectory_solution_p1 # TrajectorySolution 객체 전달
    )

    logger.print_subheader("Running P2.1 Visiting Sequence Optimization (GA)")
    opt_sequence_p2, opt_time_p2 = sequence_optimizer.optimize_sequence()
    print(f"opt_sequence_p2: {opt_sequence_p2}")
    print(f"opt_time_p2: {opt_time_p2}")
    
    trajectory_solution_p2 = TrajectorySolution(
        visiting_sequence_pi=opt_sequence_p2,
        speeds_v_mps=trajectory_solution_p1.speeds_v_mps,
        hover_positions_q_tilde_meters=trajectory_solution_p1.hover_positions_q_tilde_meters,
        hover_positions_p_meters=trajectory_solution_p1.hover_positions_p_meters,
        slacks_lambda_m=trajectory_solution_p1.slacks_lambda_m
    )
    
    # trajectory_solution_p2 = fixed_path_for_p1

    hovering_pos_optimizer = HoveringPositionOptimizerSCA(
        uav_model=uav,
        env_model=env,
        sim_config=sim_cfg,
        fixed_trajectory_params=trajectory_solution_p2 # TrajectorySolution 객체 전달
    )

    
    logger.print_subheader("Running P2.2 Hovering Position Optimization (SCA)")
    opt_q_tilde_l, opt_p_l, opt_time= hovering_pos_optimizer.optimize_hovering_positions()
    print(f"opt_q_tilde_l: {opt_q_tilde_l}")
    print(f'fixed_path_for_p1.hover_positions_q_tilde_meters: {fixed_path_for_p1.hover_positions_q_tilde_meters}')
    print(f"opt_p_l: {opt_p_l}")
    print(f'fixed_path_for_p1.hover_positions_p_meters: {fixed_path_for_p1.hover_positions_p_meters}')
    print(f"opt_time: {opt_time}")
    
    
    fixed_path_for_p1.hover_positions_p_meters = opt_p_l.copy() 
    fixed_path_for_p1.hover_positions_q_tilde_meters = opt_q_tilde_l.copy()
    
    time_energy_aois = uav.calculate_total_mission_time_energy_aois_for_trajectory(fixed_path_for_p1)
    print(f"time_energy_aois: {time_energy_aois}")
    
    
    
    

if __name__ == "__main__":
    run_experiment()
