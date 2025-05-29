import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

def plot_aoi_per_area(
    aoi_values: np.ndarray, # Expected to be a NumPy array of AoI values for each area
    save_path: Optional[str] = './results/aoi_per_area_plot.png',
):
    """
    Plots Age of Information (AoI) values versus the index of a monitoring area.
    This function aims to replicate the style of a specific academic figure,
    including an annotation for the maximum AoI value and a horizontal dashed line.

    Args:
        aoi_values: A 1D NumPy array containing the AoI values for each monitoring area.
                    The plot will display AoIs for these areas, indexed from 1.
        save_path: Optional path to save the figure. If None, the figure is not saved.
    """
    num_areas = len(aoi_values)

    if num_areas == 0:
        print("Warning: No AoI values provided. Cannot generate AoI per Area plot.")
        return

    # The reference image shows 20 areas. This function will adapt to num_areas.
    # If strictly 20 areas are required, the input `aoi_values` should have 20 elements.
    if num_areas != 20:
        print(f"Plotting AoI for {num_areas} areas. (Note: Reference image shows 20 areas).")

    area_indices = np.arange(1, num_areas + 1) # X-axis: 1, 2, ..., num_areas

    # Adjust figsize to better match the example image proportions
    plt.figure(figsize=(9, 7)) # Slightly adjusted for typical academic figure aspect

    # Main plot: AoI values vs. Area Index
    plt.plot(area_indices, aoi_values, marker='*', linestyle='-')

    # X-axis configuration
    plt.xlabel("The index of a monitoring area", fontsize=10)
    # Set x-axis ticks from 1 to num_areas. If num_areas is 20, it will be 1 to 20.
    if num_areas <= 25: # Show all integer ticks if not too many
        plt.xticks(np.arange(1, num_areas + 1, 1), fontsize=8)
    else: # If many areas, reduce tick density
        tick_step = max(1, int(np.ceil(num_areas / 20.0))) # Aim for around 20 ticks
        plt.xticks(np.arange(1, num_areas + 1, tick_step), fontsize=8)
    plt.xlim(left=0.5, right=num_areas + 0.5) # Add some padding to x-axis

    # Y-axis configuration (matching the example image style)
    plt.ylabel("The threshold of the AoI limitation (s)", fontsize=10) # Label from image
    
    min_data_val = np.min(aoi_values) if num_areas > 0 else 5
    max_data_val = np.max(aoi_values) if num_areas > 0 else 35

    # Set y-ticks to match the image style (e.g., 5, 7, 9, ..., 35)
    # Dynamically adjust based on data range while trying to maintain steps of 2.
    y_tick_min = np.floor(min_data_val / 2.0) * 2.0 
    if y_tick_min > min_data_val -1 : y_tick_min -=2 # Ensure lowest tick is at or below min_data_val
    if y_tick_min < 0 : y_tick_min = 0
    if y_tick_min % 2 != 0 and y_tick_min >0 : y_tick_min = max(0, y_tick_min -1) # Try to start from even for image style
    if y_tick_min <5: y_tick_min = 5 # Match image style start if possible

    y_tick_max = np.ceil(max_data_val / 2.0) * 2.0
    if y_tick_max < max_data_val +1 : y_tick_max +=2 # Ensure highest tick is at or above max_data_val
    if y_tick_max % 2 != 0: y_tick_max +=1
    if y_tick_max <35 and max_data_val <=35: y_tick_max = 35 # Match image style end if possible
    
    # Ensure y_tick_max is greater than y_tick_min
    if y_tick_max <= y_tick_min : y_tick_max = y_tick_min +2

    plt.title(f"AoI per Area (maximum AoI: {max_data_val:.1f}s)", fontsize=10)
    plt.yticks(np.arange(y_tick_min, y_tick_max + 1, 2), fontsize=8)
    plt.ylim(bottom=y_tick_min - 1, top=y_tick_max + 2) # Add some padding

    # Horizontal dashed line and annotation for the maximum value
    if num_areas > 0:
        max_aoi_value_in_data = np.max(aoi_values)
        max_aoi_indice = np.argmax(aoi_values)
        
        plt.axhline(y=max_aoi_value_in_data, color='k', linestyle='--', linewidth=1)

        annotation_text = f"The maximum value {max_aoi_value_in_data:.1f}"
        
        # Position the annotation text and arrow (coordinates are relative to data)
        # Text y-position: slightly above the dashed line
        text_y_coord = max_aoi_value_in_data + (plt.gca().get_ylim()[1] - max_aoi_value_in_data) * 0.1
        if text_y_coord > plt.gca().get_ylim()[1] - 1.0 : # Avoid going off top
            text_y_coord = max_aoi_value_in_data + 1.0 if max_aoi_value_in_data + 1.0 < plt.gca().get_ylim()[1] else plt.gca().get_ylim()[1] - 0.5


        # Arrow tip x-position and text x-position based on image style
        if num_areas >= 7: # Positioning similar to image if enough areas
            text_x_coord = area_indices[min(6, num_areas - 1)]   # x=7 (1-indexed) or less if fewer areas
        elif num_areas >= 2:
            text_x_coord = area_indices[0] + 0.5 * (area_indices[-1] - area_indices[0])
        else: # Single area
            text_x_coord = area_indices[0]
        
        plt.annotate(annotation_text,
                     xytext=(text_x_coord, text_y_coord+1),       # Text location
                     xy=(max_aoi_indice+1, max_aoi_value_in_data), # Arrow tip points here
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.2, headwidth=4, headlength=2),
                     fontsize=9,
                     horizontalalignment='left',
                     verticalalignment='bottom' 
                    )

    plt.grid(False) # No grid, as in the example image
    plt.tight_layout()

    if save_path:
        import os
        results_dir = os.path.dirname(save_path)
        if results_dir and not os.path.exists(results_dir): # Ensure directory exists
            os.makedirs(results_dir)
            print(f"Created directory: {results_dir}")
        plt.savefig(save_path)
        print(f"AoI per Area plot saved to {save_path}")
    
    # plt.show() # Typically called by the script that uses this function
    plt.close() # Close the figure to free up memory