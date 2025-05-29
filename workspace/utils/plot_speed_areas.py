# /path/to/your/project/plot_speed_areas.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from model.trajectory import TrajectorySolution
from model.environment import Environment

# For custom legend
from matplotlib.lines import Line2D

def plot_speed_vs_area_index(
    trajectory: 'TrajectorySolution',
    environment: 'Environment',
    save_path: Optional[str] = './results/speed_vs_area_index.png'
):
    """
    Plots the UAV's flying speed for each segment.
    X-axis shows area indices. Markers are colored by segment type:
    - Red ('r') for travel to a collection point (Q_k).
    - Blue ('b') for travel from collection (Q_k) to sending (P_k) point within an area.
    - Green ('g') for initial departure and final return.
    If dist(Q_k, P_k) < 0.1m for a 'q_to_p' segment, speed is 0.

    Args:
        trajectory: The TrajectorySolution.
        environment: The Environment object.
        save_path: Path to save the figure.
    """
    if trajectory.get_num_visited_areas() == 0:
        print("Cannot plot speeds: No areas visited in the trajectory.")
        return

    segments = trajectory.get_path_segment_details(environment)
    if not segments:
        print("Cannot plot speeds: No segments found in the trajectory.")
        return

    x_values_for_plot = [] # Will store numerical x-coordinates for plotting
    y_speeds = []
    marker_colors = []
    x_tick_positions = []
    x_tick_labels = []

    # Prepare data for plotting
    for i, segment_detail in enumerate(segments):
        x_values_for_plot.append(i) # Use segment index as the raw x-coordinate for plotting
        
        speed_idx = segment_detail['speed_idx']
        segment_type = segment_detail['segment_type']
        current_segment_speed = 0.0
        current_color = 'k' # Default color (black)

        # Determine speed
        if segment_type == 'q_to_p':
            sequence_idx_of_area = speed_idx - 1
            if 0 <= sequence_idx_of_area < trajectory.get_num_visited_areas():
                area_id_in_env = trajectory.visiting_sequence_pi[sequence_idx_of_area]
                q_k = trajectory.get_q_tilde_k_for_area_id(area_id_in_env)
                p_k = trajectory.get_p_k_for_area_id(area_id_in_env)
                if q_k is not None and p_k is not None:
                    dist_qp = np.linalg.norm(p_k - q_k)
                    if dist_qp < 0.1:
                        current_segment_speed = 0.0
                    else:
                        current_segment_speed = trajectory.speeds_v_mps[speed_idx] if speed_idx < len(trajectory.speeds_v_mps) else 0.0
                else: # q_k or p_k is None
                    current_segment_speed = trajectory.speeds_v_mps[speed_idx] if speed_idx < len(trajectory.speeds_v_mps) else 0.0
            else: # sequence_idx_of_area out of bounds
                current_segment_speed = trajectory.speeds_v_mps[speed_idx] if speed_idx < len(trajectory.speeds_v_mps) else 0.0
        else: # 'initial_to_q', 'p_to_q', 'p_to_initial'
            current_segment_speed = trajectory.speeds_v_mps[speed_idx] if speed_idx < len(trajectory.speeds_v_mps) else 0.0
        
        y_speeds.append(current_segment_speed)

        # Determine color and x-tick label based on segment type and index
        # The x-axis label refers to the *area* being processed or approached.
        # `speed_idx` helps map to area sequence index.
        # `i` is the overall segment index.
        
        num_areas_in_seq = trajectory.get_num_visited_areas()

        if segment_type == 'initial_to_q': # S0 -> Q_0 (area 0)
            current_color = 'red' # Approaching collection for area 0
            x_tick_positions.append(i)
            x_tick_labels.append(str(trajectory.visiting_sequence_pi[0])) # Label with first area ID
        elif segment_type == 'q_to_p': # Q_k -> P_k (within area k)
            current_color = 'blue' # Transmission phase for area k
            area_sequence_index = speed_idx -1 # k
            x_tick_positions.append(i)
            x_tick_labels.append(str(trajectory.visiting_sequence_pi[area_sequence_index])) # Label with area k's ID
        elif segment_type == 'p_to_q': # P_k -> Q_{k+1} (approaching area k+1)
            current_color = 'red' # Approaching collection for area k+1
            # Find which area index k+1 corresponds to in trajectory.visiting_sequence_pi
            # This segment's speed_idx is 0. We need to know which area Q_{k+1} belongs to.
            # The 'q_to_p' segment before this was for area `k`. This is for `k+1`.
            # The `segments` list is ordered. If `segments[i-1]` was a `q_to_p` for area `idx_k`,
            # then this `p_to_q` segment is heading towards area `idx_k+1`.
            # `i` is current segment index. `i-1` was Q_k -> P_k. `idx_k = segments[i-1]['speed_idx'] - 1`.
            # So this `p_to_q` is for `idx_k + 1`.
            prev_segment_speed_idx = segments[i-1]['speed_idx']
            next_area_sequence_index = prev_segment_speed_idx # Since prev was for (speed_idx-1), this is for (speed_idx)
            if next_area_sequence_index < num_areas_in_seq:
                 x_tick_positions.append(i)
                 x_tick_labels.append(str(trajectory.visiting_sequence_pi[next_area_sequence_index]))
            else: # Should not happen if segments are correctly generated
                 x_tick_positions.append(i)
                 x_tick_labels.append("Err")

        elif segment_type == 'p_to_initial': # P_N -> S0 (Return)
            current_color = 'green' # Return trip
            x_tick_positions.append(i)
            x_tick_labels.append("Ret")
        
        marker_colors.append(current_color)

    plt.figure(figsize=(10, 7)) # Adjusted figsize for better legend and label space

    # Plot the main line connecting all points (e.g., in a neutral color)
    plt.plot(x_values_for_plot, y_speeds, linestyle='-', color='grey', zorder=1)

    # Plot individual markers with specific colors
    for x_val, y_val, color in zip(x_values_for_plot, y_speeds, marker_colors):
        plt.plot(x_val, y_val, marker='*', color=color, linestyle='None', markersize=8, zorder=2)
        
    plt.xlabel("Monitoring Area Index / Phase", fontsize=12)
    plt.ylabel("The UAV's flying speed (m/s)", fontsize=12)

    # Set custom x-axis ticks
    if x_tick_positions and x_tick_labels:
        # Rotate labels if too many, to prevent overlap
        rotation_angle = 0
        tick_fontsize = 10 
        plt.xticks(ticks=x_tick_positions, labels=x_tick_labels,
                   rotation=rotation_angle, ha="right" if rotation_angle == 45 else "center",
                   fontsize=tick_fontsize)
    else: # Fallback
        max_x_tick = max(x_values_for_plot) if x_values_for_plot else 0
        plt.xticks(np.arange(0, max_x_tick + 1, 1))

    y_max_limit = 40
    padding = 0
    if max(y_speeds) > y_max_limit - 2:
        padding = 5
    plt.ylim([-2, y_max_limit + 2 + padding]) 
    plt.yticks(np.arange(0, y_max_limit + 1, 5))

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='To Collection (Qk)', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Collection to Transmission (Qk to Pk)', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Travel/Return', markerfacecolor='green', markersize=10) # For S0->Q0 and P_N->S0
    ]
    # The S0->Q0 is colored 'red' as "To Collection", so we might simplify legend or adjust coloring.
    # Let's make 'initial_to_q' red and 'p_to_initial' green. 'p_to_q' will also be red.
    # So, Red: To Collection Qk (includes S0->Q0 and Pk'->Qk)
    # Blue: Within Area (Qk->Pk)
    # Green: Return (PN->S0)

    # Adjust logic for 'initial_to_q' for green color if desired:
    # For now, 'initial_to_q' is red. 'p_to_initial' is green.
    # If S0->Q0 should be green:
    # In the loop:
    # if segment_type == 'initial_to_q':
    #    current_color = 'green'
    #    x_tick_labels.append("Start") # Or specific area 0
    # Then update legend_elements accordingly.
    # For consistency with "To Collection" being red:
    if segments[0]['segment_type'] == 'initial_to_q':
        marker_colors[0] = 'red' # Override if 'initial_to_q' was green but legend implies red
    
    # Revised legend for current coloring:
    # Red: Approaching a collection point Q_k (includes S0->Q_0 and P_k' -> Q_k)
    # Blue: Moving from Q_k to P_k (within an area for transmission)
    # Green: Final return P_N -> S0
    
    # Recheck color logic for `initial_to_q` and `p_to_q`
    # `initial_to_q` -> red (approaching first Q)
    # `p_to_q` -> red (approaching next Q)
    # `q_to_p` -> blue (within area)
    # `p_to_initial` -> green (return)
    
    # This matches the legend below:
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Approach Collection (Qk)', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Collection to Transmission (Qk to Pk)', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Return to Start', markerfacecolor='green', markersize=10)
    ]

    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.25)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for a potential title if added later
    plt.title(f"{save_path.split('_')[-2].upper()}: UAV's Flying Speed for each Area Index", fontsize=12)

    if save_path:
        import os
        results_dir = os.path.dirname(save_path)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)
        plt.savefig(save_path)
        print(f"Speed vs. Area Index plot saved to {save_path}")
    
    plt.close()