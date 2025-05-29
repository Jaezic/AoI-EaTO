# workspace/configs/data_classes.py

from dataclasses import dataclass, field
# from typing import List, Dict, Optional
from typing import List, Optional, Tuple, Dict

# --- UAV Params ---
@dataclass
class SpeedParams:
    max_vmax_mps: float
    min_init_vmin_mps: float

@dataclass
class PropulsionPowerModelParams:
    P0_watts: float
    P1_watts: float
    v_tip_mps: float
    v0_hover_mps: float
    d0_ratio: float
    rho_kg_per_m3: float
    s_solidity: float
    A_disc_m2: float

@dataclass
class EnergyParams:
    max_elimit_joule: float
    propulsion_power_model: PropulsionPowerModelParams

@dataclass
class LosParams:
    a: float
    b: float

@dataclass
class CommunicationParams:
    pu_transmit_power_dbm: float
    beta0_db: float
    noise_power_sigma2_dbm: float
    snr_gap_gamma_db: float
    path_loss_alpha: float
    los_params: LosParams
    kappa_nlos: float
    plos_approx_probability: float
    snr_min_db: float
    d_min_comm_meters: float
    channel_bandwidth_B_mhz: float
    data_packet_size_Sk_mbits: float

@dataclass
class SensingParams:
    zeta_parameter: float
    p_th_probability: float
    t_int_seconds: float
    uav_monitoring_range_ru_meters: float
    data_collection_time_te_seconds: float

@dataclass
class UAVConfig:
    speed: SpeedParams
    energy: EnergyParams
    sensing: SensingParams
    communication: CommunicationParams
    altitude_hu_meters: float

# --- Environment Params ---
@dataclass
class EnvironmentConfig:
    num_areas: int
    area_positions_wk_meters: List[List[float]] # each area's [x, y] coordinates list
    area_radius_ra_meters: float                # assume the same radius for all areas
    initial_uav_position_s0_meters: List[float] # [x, y]
    gbs_altitude_hb_meters: float               # GBS altitude (H_B)
    # optionally, set the entire GBS position
    gbs_position_meters: Optional[List[float]] = None # [x, y, z], if None, use (0,0,H_B)

# --- Simulation Params ---
@dataclass
class AoiEatoParams:
    max_iterations: int
    convergence_threshold_seconds: float

@dataclass
class GAParams:
    population_size: int
    num_generations: int
    mutation_rate: float
    crossover_rate: float
    selection_pressure_ratio: float
    aoi_tolerance: float
    energy_tolerance: float

@dataclass
class GeneralSimParams:
    # aoi_limit_seconds: float
    # random_seed: int = None # if None, do not set the seed
    aoi_limit_seconds: Optional[float] = 70.0 # default global AoI limit
    aoi_limit_seconds_per_poi: Optional[List[float]] = None # POI-specific AoI limit list
    random_seed: Optional[int] = 42
    
@dataclass
class SimulationConfig:
    aoi_eato: AoiEatoParams
    ga: GAParams
    general: GeneralSimParams

@dataclass
class ExperimentFixedPathParams:
    """for P1.2 test, set the fixed path and initial values"""
    fixed_visiting_sequence: List[int]
    initial_speeds_mps: List[float]
    fixed_hover_q_tilde_meters: List[List[float]]
    fixed_hover_p_meters: List[List[float]]
    initial_slacks_lambda: List[float]