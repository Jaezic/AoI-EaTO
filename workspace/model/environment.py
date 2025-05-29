# model/environment.py
import numpy as np
from typing import List, Tuple, Optional

# use the data classes in the params folder (the actual path is adjusted to match the project structure)
from configs.data_classes import EnvironmentConfig, SimulationConfig 

class MonitoringArea:
    """
    A data class representing the information of a single monitoring area.
    Corresponds to A_k in the paper.
    """
    id: int                          # unique ID of the monitoring area
    position_wk_meters: np.ndarray   # center position (x, y) coordinates, w_k in the paper
    radius_ra_meters: float          # radius of the monitoring area, r_a in the paper

    def __init__(self, area_id: int, position_wk_meters: Tuple[float, float], radius_ra_meters: float):
        """
        Initialize a MonitoringArea object.

        Args:
            area_id (int): area ID.
            position_wk_meters (Tuple[float, float]): center position (x, y) coordinates (meters).
            radius_ra_meters (float): radius of the monitoring area (meters).
        """
        if not isinstance(area_id, int) or area_id < 0:
            raise ValueError("Area ID must be a non-negative integer.")
        if not (isinstance(position_wk_meters, (list, tuple, np.ndarray)) and len(position_wk_meters) == 2):
            raise ValueError("Position must be a 2-element tuple, list, or numpy array.")
        if not isinstance(radius_ra_meters, (int, float)) or radius_ra_meters <= 0:
            raise ValueError("Radius must be a positive number.")

        self.id = area_id
        self.position_wk_meters = np.array(position_wk_meters, dtype=float)
        self.radius_ra_meters = float(radius_ra_meters)

    def __repr__(self) -> str:
        return f"MonitoringArea(id={self.id}, pos={self.position_wk_meters}, radius={self.radius_ra_meters})"


class GBS:
    """
    A data class representing the information of a Ground Base Station (GBS).
    """
    # The horizontal position of the GBS is assumed to be (0,0), and only the altitude is managed, which is consistent with the paper.
    # Alternatively, the entire 3D position can be managed flexibly.
    position_meters: np.ndarray  # GBS's position (x, y, z) coordinates.
    altitude_hb_meters: float    # GBS's altitude, H_B in the paper (same as position_meters[2])

    def __init__(self, position_meters: Tuple[float, float, float]):
        """
        Initialize a GBS object.

        Args:
            position_meters (Tuple[float, float, float]): GBS's position (x, y, z) coordinates (meters).
                                                            According to the paper, x and y may be fixed to 0.
        """
        if not (isinstance(position_meters, (list, tuple, np.ndarray)) and len(position_meters) == 3):
            raise ValueError("GBS position must be a 3-element tuple, list, or numpy array.")

        self.position_meters = np.array(position_meters, dtype=float)
        self.altitude_hb_meters = self.position_meters[2]

    def get_horizontal_position_meters(self) -> np.ndarray:
        """return the horizontal (x, y) position of the GBS"""
        return self.position_meters[:2]

    def __repr__(self) -> str:
        return f"GBS(pos={self.position_meters}, altitude_hb={self.altitude_hb_meters})"


class Environment:
    """
    A data class representing the entire simulation environment.
    It includes information about the monitoring areas, GBS, initial UAV position, etc.
    """
    monitoring_areas: List[MonitoringArea]
    gbs: GBS
    initial_uav_position_s0_meters: np.ndarray # initial ground projection position of the UAV (x, y)
    num_areas: int
    config: EnvironmentConfig
    env_config: EnvironmentConfig 
    sim_config: Optional[SimulationConfig] 

    def __init__(self, config: EnvironmentConfig, sim_config: Optional[SimulationConfig] = None):
        """
        Initialize an Environment object.
        Use the environment configuration loaded from YAML, etc.

        Args:
            config (EnvironmentConfig): a data class object containing environment configuration values.
        """
        if not isinstance(config, EnvironmentConfig):
            raise TypeError("config must be an instance of EnvironmentConfig.")

        self.config = config
        self.sim_config = sim_config 
        self.num_areas = config.num_areas
        if len(config.area_positions_wk_meters) != self.num_areas:
            raise ValueError(f"Number of area positions ({len(config.area_positions_wk_meters)}) "
                             f"does not match num_areas ({self.num_areas}).")

        self.monitoring_areas = []
        for i in range(self.num_areas):
            # assume that there is only one value for area_radius_ra_meters in EnvironmentConfig and it is applied to all areas
            # or there is a list of area_configs, then use config.area_configs[i].radius_ra_meters
            area = MonitoringArea(
                area_id=i, # assign ID from 0 to N-1 sequentially
                position_wk_meters=tuple(config.area_positions_wk_meters[i]), # convert to tuple
                radius_ra_meters=config.area_radius_ra_meters
            )
            self.monitoring_areas.append(area)

        # GBS position is directly obtained from config, or (0,0,H_B) form
        # here, assume that config.gbs_position_meters exists (x,y,z)
        # or only config.gbs_altitude_hb_meters exists, then create (0,0,alt)
        if hasattr(config, 'gbs_position_meters') and config.gbs_position_meters is not None:
             gbs_pos = tuple(config.gbs_position_meters)
        else: # if only config.gbs_altitude_hb_meters exists, then create (0,0,alt)
             gbs_pos = (0.0, 0.0, config.gbs_altitude_hb_meters)
        self.gbs = GBS(position_meters=gbs_pos)


        if not (isinstance(config.initial_uav_position_s0_meters, (list, tuple, np.ndarray)) and \
                len(config.initial_uav_position_s0_meters) == 2):
            raise ValueError("Initial UAV position s0 must be a 2-element list, tuple or numpy array.")
        self.initial_uav_position_s0_meters = np.array(config.initial_uav_position_s0_meters, dtype=float)

    def get_current_aoi_limit_seconds(self) -> Optional[float]:
        """ 
        Return the current applied AoI limit time (seconds).
        If sim_config exists and aoi_limit_seconds is defined in it, return that value.
        """
        if self.sim_config and hasattr(self.sim_config, 'general') and \
           hasattr(self.sim_config.general, 'aoi_limit_seconds'):
            return self.sim_config.general.aoi_limit_seconds
        return None # AoI limit is not set or unknown
    
    def get_aoi_limit_for_poi(self, poi_id: int) -> Optional[float]:
        """Return the AoI limit time applied to a specific POI."""
        if self.sim_config and hasattr(self.sim_config, 'general'):
            if self.sim_config.general.aoi_limit_seconds_per_poi and \
               0 <= poi_id < len(self.sim_config.general.aoi_limit_seconds_per_poi):
                return self.sim_config.general.aoi_limit_seconds_per_poi[poi_id]
            elif self.sim_config.general.aoi_limit_seconds is not None:
                return self.sim_config.general.aoi_limit_seconds
        return None # default value or no setting

    def get_area_by_id(self, area_id: int) -> Optional[MonitoringArea]:
        """
        Return the MonitoringArea object corresponding to the given ID.
        Assume that MonitoringArea objects are stored in a list in order of ID.

        Args:
            area_id (int): the ID of the monitoring area to find (0 to N-1).

        Returns:
            MonitoringArea | None: the MonitoringArea object corresponding to the given ID, or None (if the ID is out of range).
        """
        if 0 <= area_id < len(self.monitoring_areas):
            # assume that the ID and list index match
            if self.monitoring_areas[area_id].id == area_id:
                return self.monitoring_areas[area_id]
            else: # if the ID and index do not match, search by ID
                for area in self.monitoring_areas:
                    if area.id == area_id:
                        return area
        return None

    def get_all_area_positions_wk(self) -> np.ndarray:
        """
        Return the center positions (w_k) of all monitoring areas as a numpy array.
        The order of the array is the same as the area_id order.

        Returns:
            np.ndarray: an array of shape (N, 2), each row is the (x, y) coordinates of a monitoring area.
        """
        if not self.monitoring_areas:
            return np.array([[]], dtype=float) # return an empty array
        return np.array([area.position_wk_meters for area in self.monitoring_areas])

    def get_total_monitoring_areas(self) -> int:
        """Return the total number of monitoring areas."""
        return self.num_areas

    def __repr__(self) -> str: # __repr__ also updated the properties
        aoi_limit_info = f", aoi_limit={self.get_current_aoi_limit_seconds()}s" if self.get_current_aoi_limit_seconds() is not None else ""
        return (f"Environment(num_areas={self.num_areas}, "
                f"gbs={self.gbs}, "
                f"initial_uav_pos={self.initial_uav_position_s0_meters}, "
                f"monitoring_areas_count={len(self.monitoring_areas)}{aoi_limit_info})")

    # def __repr__(self) -> str:
    #     return (f"Environment(num_areas={self.num_areas}, "
    #             f"gbs={self.gbs}, "
    #             f"initial_uav_pos={self.initial_uav_position_s0_meters}, "
    #             f"monitoring_areas_count={len(self.monitoring_areas)})")