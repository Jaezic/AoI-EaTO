import yaml
from pathlib import Path
from dacite import from_dict
from .data_classes import UAVConfig, EnvironmentConfig, SimulationConfig, ExperimentFixedPathParams

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "params" # the root of the configs folder

def load_yaml_config(file_path: Path, data_class: type):
    try:
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return from_dict(data_class=data_class, data=config_data)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {file_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        raise
    except Exception as e:
        print(f"Error creating data class from YAML {file_path}: {e}")
        raise

def get_uav_config() -> UAVConfig:
    return load_yaml_config(CONFIG_DIR / "uav_params.yaml", UAVConfig)

def get_environment_config() -> EnvironmentConfig:
    return load_yaml_config(CONFIG_DIR / "environment_params.yaml", EnvironmentConfig)

def get_simulation_config() -> SimulationConfig:
    return load_yaml_config(CONFIG_DIR / "simulation_params.yaml", SimulationConfig)

def get_experiment_fixed_path_params() -> ExperimentFixedPathParams:
    """Load the fixed path parameters for P1.2 test."""
    return load_yaml_config(CONFIG_DIR / "experiment_params.yaml", ExperimentFixedPathParams)