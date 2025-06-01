# AoI-EATO: UAV Trajectory Optimization for Time-Constrained Data Collection in UAV-Enabled Environmental Monitoring Systems

This repository provides the first unofficial implementation of the paper "UAV Trajectory Optimization for Time-Constrained Data Collection in UAV-Enabled Environmental Monitoring Systems" (IEEE Internet of Things Journal 2022). You can find the paper on [IEEE Xplore](https://ieeexplore.ieee.org/document/9817083).

## Overview

![overview](./figs/overview.jpg)

This project focuses on optimizing the trajectory of an Unmanned Aerial Vehicle (UAV) dispatched to a geographical area for collecting data from a set of **monitoring areas**. The collected data is time-constrained and needs to be transmitted to a Ground Base Station (GBS). The primary goal is to **minimize the UAV's overall mission completion time**. This optimization jointly considers the UAV's flying speeds, hovering positions (for data collection and transmission), and visiting sequence, while adhering to constraints on the **Age of Information (AoI)** of data from each monitoring area and the UAV's **on-board energy limitations**.

The core of this implementation is the **AoI-EaTO (Age of Information - and Energy-aware Trajectory Optimization)** algorithm, as proposed in the paper. This algorithm decomposes the main problem and iteratively optimizes:
1.  **UAV Speed Optimization**: Using a Successive Convex Approximation (SCA) method-based algorithm.
2.  **UAV Path Optimization**:
    *   **Visiting Sequence**: Using a Genetic Algorithm (GA)-based algorithm.
    *   **Hovering Positions**: Using an SCA method-based algorithm for both data collection (`q̃_k`) and data transmission (`p_k`) hovering spots.

The project also includes implementations for baseline comparison algorithms, such as "Random" visiting sequence and "Greedy" visiting sequence, to evaluate the performance of the AoI-EaTO algorithm.

## Setup & Getting Started

### Prerequisites
*   Docker (if you want)
*   Python 3.11 (as specified in Dockerfile)

### Dependencies
The main Python dependencies are listed in `requirements.txt`:
*   numpy
*   dacite
*   pyyaml
*   cvxpy
*   matplotlib
*   ecos

### Running with Docker
This is the recommended way to run the simulation.

1.  **Build the Docker image and run the container:**
    ```bash
    docker-compose up --build -d
    ```
2.  **Access the running container:**
    ```bash
    docker exec -it uav_aoi_eto_project /bin/bash
    ```
3.  **Navigate to the workspace and run the simulation:**
    Inside the container's shell:
    ```bash
    cd /workspace
    python main.py --algorithm [ALGORITHM_NAME]
    ```
    Replace `[ALGORITHM_NAME]` with one of the implemented algorithms. Based on `main.py`, possible values are:
    *   `AoI-EaTO`
    *   `Random`
    *   `Greedy`

    For example, to run the AoI-EaTO algorithm:
    ```bash
    python main.py --algorithm AoI-EaTO
    ```
    *(Note: The exact command-line arguments for `main.py` might need verification by checking the `if __name__ == "__main__":` block in `workspace/main.py`. The above is a common way to pass algorithm choices.)*

### Running Locally (Alternative)

1.  **Ensure Python 3.11 is installed.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Navigate to the workspace directory and run the simulation:**
    ```bash
    cd workspace
    python main.py --algorithm [ALGORITHM_NAME]
    ```
    (Same `main.py` execution notes apply as in the Docker section).

### Run Energy Limitation & AoI Limitation Experiments
1.  **Energy Limitation Experiment**
    ```bash
    cd workspace
    python utils/plot_fig8_9.py
    ```
1.  **AoI Limitation Experiment**
    ```bash
    cd workspace
    python utils/plot_fig13.py
    ```
## Results

![image](./figs/image.png)
![image2](./figs/image2.png)
![image3](./figs/image3.png)
![image4](./figs/image4.png)
![image5](./figs/image5.png)
![image6](./figs/image6.png)

The simulation results aim to demonstrate the effectiveness of the AoI-EaTO algorithm in finding an optimized UAV trajectory that minimizes mission completion time while satisfying AoI and energy constraints. Key outcomes from the simulation, mirroring the paper's findings, include:
*   An optimized **visiting sequence** (`π*`) for the UAV to service the monitoring areas.
*   Calculated optimal **flying speeds** (`v_m^*`) for different segments of the UAV's path.
*   Determined optimal **hovering locations** for data collection (`q̃_k^*`) and data transmission (`p_k^*`).
*   Detailed logs and outputs quantifying the **mission completion time (T)**, **total energy consumption (E_all)**, and the **maximum AoI (AoI_max^k)** for data from each monitoring area.
*   Visualizations of the UAV trajectory, hovering positions, GBS location, monitoring areas, and potentially communication/sensing ranges, similar to figures presented in the paper.
*   Analysis of how the **AoI limitation threshold** and the **UAV's on-board energy** impact the achievable mission performance.

*(This section can be expanded with specific graphs, figures, or key performance indicators from your experiments once available. The `results/` directory is intended for storing such outputs, like trajectory plots.)*
