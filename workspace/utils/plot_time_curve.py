import matplotlib.pyplot as plt

def plot_time_curve(times_list: list, save_path: str = './results/time_curve.png'):
    plot_iteration_numbers = list(range(len(times_list)))
    plt.figure(figsize=(10, 6)) # You can adjust figure size
    plt.plot(plot_iteration_numbers, times_list, marker='*', linestyle='-')
    plt.xlabel("The number of iterations")
    plt.ylabel("The mission completion time (s)")
    plt.title("AoI-EaTO: Mission Completion Time vs Iterations")
    if plot_iteration_numbers: # Ensure list is not empty
        # Set x-axis ticks to be integers from 1 to the max iteration number plotted
        plt.xticks(range(1, max(plot_iteration_numbers) + 1))
        
        # Optional: Adjust y-axis limits to be similar to the example plot or based on data
        min_y_val = min(times_list)
        max_y_val = max(times_list)
        y_padding = (max_y_val - min_y_val) * 0.1 # 10% padding
        plt.ylim(max(0, min_y_val - y_padding), max_y_val + y_padding)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


if __name__ == "__main__":
    plot_time_curve([1, 2, 3, 4, 5])