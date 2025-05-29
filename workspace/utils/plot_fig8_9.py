# utils/plot_fig8.py

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

def calculate_average_flying_speed(trajectory):
    """
    Calculate the average flying speed from the given trajectory object.
    Assume that trajectory.speeds_v_mps is the array of speeds in the flight section.
    """
    if trajectory and hasattr(trajectory, 'speeds_v_mps') and \
       trajectory.speeds_v_mps is not None and len(trajectory.speeds_v_mps) > 0:
        
        return np.mean(trajectory.speeds_v_mps)
    return np.nan

def generate_fig8_9_plot():
    """
    Generate a plot similar to Fig. 8: The average flying speed of the UAV vs. the energy limitation.
    """
    energy_limits_kj = np.arange(250, 371, 10)

    algorithms_to_run = ["AoI-EaTO", "Greedy", "Random"]
    
    plot_labels = {
        "AoI-EaTO": "AoI-EaTO algorithm",
        "Greedy": "Greedy-based algorithm",
        "Random": "Random-based algorithm"
    }
    plot_colors = {
        "AoI-EaTO": "#0072BD",
        "Greedy": "#D95319",  
        "Random": "#EDB120"  
    }

    average_flying_speeds = {alg: [] for alg in algorithms_to_run}
    mission_completion_times = {alg: [] for alg in algorithms_to_run}
    
    total_runs = len(energy_limits_kj) * len(algorithms_to_run)
    current_run = 0

    print(f"Fig. 8 and Fig. 9 are generated with {total_runs} simulations...")

    results_dir_for_figs = os.path.join(project_root, "results")
    if not os.path.exists(results_dir_for_figs):
        os.makedirs(results_dir_for_figs)
        print(f"Created directory for plots: {results_dir_for_figs}")

    for energy_kj in energy_limits_kj:
        print(f"\n--- Energy limitation: {energy_kj} KJ ---")
        for alg_name in algorithms_to_run:
            current_run += 1
            start_time_alg = time.time()
            print(f"  ({current_run}/{total_runs}) Running: {alg_name}...")

            trajectory, uav, env = run_experiment(algorithm_to_run=alg_name, energy_limit_kj=energy_kj)

            if trajectory and uav:
                final_time, final_energy, _ = uav.calculate_total_mission_time_energy_aois_for_trajectory(trajectory)
                
                avg_speed = calculate_average_flying_speed(trajectory)
                average_flying_speeds[alg_name].append(avg_speed)
                mission_completion_times[alg_name].append(final_time)
                if not np.isnan(avg_speed):
                    print(f"    {alg_name} completed. Average speed: {avg_speed:.2f} m/s, Consumed energy: {final_energy/1000.0:.2f}KJ (limit: {energy_kj}KJ), Mission completion time: {final_time:.2f}s")
                else:
                    print(f"    {alg_name} completed (average speed calculation failed). Consumed energy: {final_energy/1000.0:.2f}KJ (limit: {energy_kj}KJ), Mission completion time: {final_time:.2f}s")
            else:
                average_flying_speeds[alg_name].append(np.nan)
                mission_completion_times[alg_name].append(np.nan)
                print(f"    {alg_name} failed (run_experiment did not return valid results).")
            
            end_time_alg = time.time()
            print(f"    {alg_name} execution time: {end_time_alg - start_time_alg:.2f} seconds")

    print("\nAll simulations completed. Plotting Fig. 8 and Fig. 9 results...")

    fig8_save_path = plotting_results(energy_limits_kj, algorithms_to_run, plot_labels, plot_colors, average_flying_speeds, results_dir_for_figs, fig_type="fig8")
    fig9_save_path = plotting_results(energy_limits_kj, algorithms_to_run, plot_labels, plot_colors, mission_completion_times, results_dir_for_figs, fig_type="fig9")
    print(f"\nFig. 8 style plot saved at: {fig8_save_path}")
    print(f"\nFig. 9 style plot saved at: {fig9_save_path}")

def plotting_results(energy_limits_kj, algorithms_to_run, plot_labels, plot_colors, data_dict, results_dir_for_figs, fig_type="fig8"):
    if fig_type != "fig8" and fig_type != "fig9":
        raise ValueError("fig_type must be either 'fig8' or 'fig9'")

    fig, ax = plt.subplots(figsize=(12, 7))

    num_energy_levels = len(energy_limits_kj)
    num_algorithms = len(algorithms_to_run)
    
    x_indices = np.arange(num_energy_levels)
    bar_width = 0.25 
    offsets = np.linspace(-bar_width * (num_algorithms -1) / 2, bar_width * (num_algorithms -1) / 2, num_algorithms)

    for i, alg_name in enumerate(algorithms_to_run):
        datas = data_dict[alg_name]
        ax.bar(x_indices + offsets[i], datas, bar_width, label=plot_labels[alg_name], color=plot_colors[alg_name])

    ax.set_xlabel("The energy limitation (KJ)", fontsize=12)
    if fig_type == "fig8":
        ax.set_ylabel("The UAV's average flying speed (m/s)", fontsize=12)
    elif fig_type == "fig9":
        ax.set_ylabel("The minimum mission completion time (s)", fontsize=12)
    
    if fig_type == "fig8":
        ax.set_ylim(20, 45)
        ax.set_yticks(np.arange(20, 46, 5))
    elif fig_type == "fig9":
        ax.set_ylim(600, 1100)
        ax.set_yticks(np.arange(600, 1101, 50))

    ax.set_xticks(x_indices)
    ax.set_xticklabels(energy_limits_kj)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()

    fig_save_path = os.path.join(results_dir_for_figs, f"{fig_type}_results_vs_energy.png")
    plt.savefig(fig_save_path)
    plt.close()
    return fig_save_path


if __name__ == "__main__":
    overall_start_time = time.time()
    generate_fig8_9_plot()
    overall_end_time = time.time()
    print(f"총 실행 시간: {(overall_end_time - overall_start_time)/60.0:.2f} 분")