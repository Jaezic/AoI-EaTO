# utils/plot_fig8.py

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from main import run_experiment
    print("Successfully imported run_experiment from main.py")
except ImportError as e:
    print(f"Error importing run_experiment: {e}")
    print(f"Ensure main.py is in the project root directory: {project_root}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

def generate_fig13_plot():
    """
    create a plot similar to Fig. 13: the average flying speed of the UAV vs. the energy limitation
    """
    aoi_limits = np.arange(35, 46, 1)

    algorithms_to_run = ["AoI-EaTO"]
    
    # dictionary to store the results
    mission_completion_times = {alg: [] for alg in algorithms_to_run}
    
    total_runs = len(aoi_limits) * len(algorithms_to_run)
    current_run = 0

    print(f"Start running {total_runs} simulations for Fig. 13...")

    # create a directory for the results
    results_dir_for_figs = os.path.join(project_root, "results")
    if not os.path.exists(results_dir_for_figs):
        os.makedirs(results_dir_for_figs)
        print(f"Created directory for plots: {results_dir_for_figs}")

    for aoi_limit in aoi_limits:
        print(f"\n--- AOI limit: {aoi_limit} ---")
        for alg_name in algorithms_to_run:
            current_run += 1
            start_time_alg = time.time()
            print(f"  ({current_run}/{total_runs}) Running: {alg_name}...")

            # use the run_experiment function in main.py
            # run_experiment returns (None, None, None) if it fails
            trajectory, uav, env = run_experiment(algorithm_to_run=alg_name, aoi_limit_s=aoi_limit, plot=False)

            if trajectory and uav: # if the trajectory and uav objects are received successfully
                # calculate the mission completion time and energy
                final_time, final_energy, _ = uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)
                
                # check the energy constraint (with a small error)
                mission_completion_times[alg_name].append(final_time)
            else:
                # simulation failed (run_experiment returns None, None, None)
                mission_completion_times[alg_name].append(np.nan)
                print(f"    {alg_name} failed (run_experiment returns None, None, None).")
            
            end_time_alg = time.time()
            print(f"    {alg_name} execution time: {end_time_alg - start_time_alg:.2f} seconds")

    print("\nAll simulations completed. Plotting Fig. 13 results...")

    # --- create the final bar graph ---
    fig13_save_path = plot_aoi_limit_vs_mission_time(aoi_limits, np.array(mission_completion_times[algorithms_to_run[0]]))
    print(f"\nFig. 13 results are saved in the following path: {fig13_save_path}")


def plot_aoi_limit_vs_mission_time(
    aoi_limits: np.ndarray,           # X-axis values (e.g., np.arange(28, 41, 1))
    mission_times: np.ndarray,      # Y-axis values (corresponding mission times)
    algorithm_name: str = "AoI-EaTO", # Name for potential use in title or legend
    save_path: Optional[str] = './results/fig13_results_vs_aoi_limit.png',
):
    """
    Creates a line plot of minimum mission completion time versus AoI limitation,
    styled similarly to the provided reference image.

    Args:
        aoi_limits: 1D NumPy array of AoI limitation values (x-axis).
        mission_times: 1D NumPy array of corresponding mission completion times (y-axis).
        algorithm_name: Name of the algorithm (for context, not directly used in plot labels from image).
        save_path: Optional path to save the figure.
        show_plot: Whether to display the plot.
    """
    if len(aoi_limits) != len(mission_times):
        print(f"Error: aoi_limits and mission_times must have the same length., len(aoi_limits): {len(aoi_limits)}, len(mission_times): {len(mission_times)}")
        return
    if len(aoi_limits) == 0:
        print("Error: No data provided for plotting.")
        return

    plt.figure(figsize=(10, 8)) # Adjust as needed, image looks somewhat standard

    plt.plot(aoi_limits, mission_times, marker='*', linestyle='-', color='#0072BD') # Standard blue

    # X-axis configuration
    plt.xlabel("The AoI limitation (s)", fontsize=12)
    plt.xticks(aoi_limits, fontsize=10) # Show all provided AoI limits as ticks
    # Pad x-axis slightly if desired
    plt.xlim(min(aoi_limits) - 0.5, max(aoi_limits) + 0.5)


    # Y-axis configuration
    plt.ylabel("The minimum mission completion time (s)", fontsize=12)


    plt.grid(False) # Match image

    plt.title(f"Mission Time vs. AoI Limitation for {algorithm_name}", fontsize=14) # Optional title
    plt.tight_layout()

    if save_path:
        # Ensure directory exists
        results_dir_for_plot = os.path.dirname(save_path)
        if results_dir_for_plot and not os.path.exists(results_dir_for_plot):
            os.makedirs(results_dir_for_plot)
            print(f"Created directory: {results_dir_for_plot}")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()
    return save_path


if __name__ == "__main__":
    overall_start_time = time.time()
    generate_fig13_plot()
    overall_end_time = time.time()
    print(f"Total execution time: {(overall_end_time - overall_start_time)/60.0:.2f} minutes")