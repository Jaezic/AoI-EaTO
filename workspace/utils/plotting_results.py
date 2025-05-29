import matplotlib.pyplot as plt
import numpy as np
from model.trajectory import TrajectorySolution
from model.environment import Environment
from model.uav import UAV


def plot_trajectory(trajectory: TrajectorySolution,
                    env: Environment,
                    uav: UAV,
                    show: bool = True,
                    save_path: str = None):
    """
    Plot the UAV trajectory, including monitoring areas, data collection/transmission points,
    base station, flight path, and allowed communication/area boundaries.

    Args:
        trajectory (TrajectorySolution): Optimized trajectory solution.
        env (Environment): Environment model containing area and GBS info.
        uav (UAV): UAV model for distance/rate computations.
        show (bool): Whether to display the plot interactively.
        save_path (str): If provided, save the figure to this path.

    Returns:
        fig, ax: Matplotlib figure and axes.
    """
    # Extract data
    centers = np.array(env.config.area_positions_wk_meters)
    ra = env.config.area_radius_ra_meters
    q_all = trajectory.hover_positions_q_tilde_meters
    p_all = trajectory.hover_positions_p_meters
    seq = trajectory.visiting_sequence_pi
    s0 = env.initial_uav_position_s0_meters
    bs_pos = env.gbs.get_horizontal_position_meters()

    # Compute boundaries
    # Geographical area radius: max distance of any area center + its radius
    geo_radius = np.max(np.linalg.norm(centers, axis=1)) + ra
    # Maximum allowed transmission horizontal distance
    d_max = uav.get_max_communication_horizontal_distance_dmax_meters_13()

    # Create plot
    fig, ax = plt.subplots(figsize=(10,10))

    # Monitoring areas
    for i, center in enumerate(centers):
        circle = plt.Circle(center, ra, facecolor='none', edgecolor='orange', linestyle='-', label='Monitoring area position' if i==0 else None)
        ax.add_patch(circle)
        ax.text(center[0]+35, center[1], str(i), color='black', fontsize=9, ha='center', va='center')

    # Data collection and transmission positions
    for idx, k in enumerate(seq):
        ax.plot(q_all[k,0], q_all[k,1], marker='x', linestyle='None', markersize=8, color='purple', label='Data collection position' if idx==0 else None)
        ax.plot(p_all[k,0], p_all[k,1], marker='*', linestyle='None', markersize=8, color='green', label='Data transmission position' if idx==0 else None)

    # Base station
    ax.plot(bs_pos[0], bs_pos[1], marker='^', markersize=12, color='green', label='BS')

    # UAV flight path
    # Build flight sequence: s0 -> q1 -> p1 -> q2 -> p2 -> ... -> s0
    waypoints = [s0]
    for k in seq:
        waypoints.append(q_all[k])
        waypoints.append(p_all[k])
    waypoints.append(s0)
    waypoints = np.array(waypoints)
    ax.plot(waypoints[1:,0], waypoints[1:,1], '-', color='skyblue', linewidth=1.5, label='UAV path')
    dx = waypoints[1,0] - waypoints[0,0]
    dy = waypoints[1,1] - waypoints[0,1]
    head_width = 30
    head_length = 60
    ax.arrow(waypoints[0,0], waypoints[0,1], dx, dy, 
            head_width=head_width, head_length=head_length, 
            fc='red', ec='red', length_includes_head=True)
    ax.plot(waypoints[0,0], waypoints[0,1], marker='o', markersize=8, color='red', label='Start')
    ax.text(waypoints[0,0]+60, waypoints[0,1], 'Start', color='black', fontsize=9, ha='center', va='center')

    # # Start arrow
    # if len(waypoints) > 1:
    #     start_pt = waypoints[0]
    #     next_pt = waypoints[1]
    #     dx, dy = next_pt - start_pt
    #     ax.annotate('start', xy=start_pt, xytext=(start_pt[0]+dx*0.1, start_pt[1]+dy*0.1),
    #                 arrowprops=dict(arrowstyle='->', color='red'), color='red')

    # Communication distance boundary
    dmax_circle = plt.Circle((0,0), d_max, facecolor='none', edgecolor='green', linestyle='--', label='Maximum allowed transmission distance')
    ax.add_patch(dmax_circle)

    # Geographical area boundary
    geo_circle = plt.Circle((0,0), geo_radius, facecolor='none', edgecolor='blue', linestyle=':', label='Geographical area')
    ax.add_patch(geo_circle)

    # Labels and legend
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')

    final_time, final_energy, final_aois = uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)
    ax.set_title(f"Iter {save_path.split('_')[-1].split('.')[0]} Mission Time: {final_time:.2f}s, Energy: {final_energy:.2f}J, AoIs: {final_aois.mean():.2f}, Speed: {np.mean(trajectory.speeds_v_mps):.2f}m/s")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # if show:
    #     plt.show()
    return fig, ax
