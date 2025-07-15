# Artificial Potential Field + Dijkstra Path Planning

This repository demonstrates the implementation of a hybrid Artificial Potential Field (APF) algorithm combined with Dijkstra path planning for robust path planning in a 2D environment with obstacles.

Key features include:

* **Hybrid APF + Dijkstra Implementation:** Combines local Artificial Potential Field navigation with global Dijkstra path planning for robust obstacle avoidance.
* **Standard APF Implementation:** Uses attractive forces towards the goal and repulsive forces from obstacles based on standard potential field formulas.
* **Global Path Planning:** Integrates Dijkstra algorithm on a discrete grid to provide global guidance and avoid local minima.
* **Configurable Parameters:** Allows tuning of attractive gain (`xi`), repulsive gain (`eta`), attractive threshold (`sigma0`), repulsive influence radius (`rho0`), and circumferential gain (`k_circ`) for escaping local minima.
* **Hybrid Planning:** Combines the local APF with the global Dijkstra path planner to guide the agent and mitigate local minima issues.
* **Momentum:** Incorporates velocity and momentum into the simulation loop for smoother movement.
* **Visualization:** Generates plots showing the APF force vector field, potential contour map, 3D potential surface, Dijkstra path, obstacle locations, start/goal points, and the agent's final path.

![image](https://github.com/user-attachments/assets/4f048e46-8924-4faa-aba0-589f82d56146)

## Installation

You can install this module in your project using:

```bash
pip install path/to/this/project
```

Or, if published to PyPI:

```bash
pip install artificial-potential-field-example
```

## Usage

### Quick Demo

Try the built-in demos to see the APF + Dijkstra module in action:

```python
from artificial_potential_field_example import demo_basic_usage, demo_comprehensive

# Run basic demo showing core APF + Dijkstra features
demo_basic_usage()

# Run comprehensive demo showing all APF + Dijkstra features
demo_comprehensive()
```

### Simple Usage with APFSimulator

The easiest way to use the APF + Dijkstra module is with the `APFSimulator` class:

```python
from artificial_potential_field_example import APFSimulator, generate_random_obstacles

# Define start and goal
start = (0, 0)
goal = (40, 40)

# Generate random obstacles
obstacles = generate_random_obstacles(start, goal, num_obstacles=15)

# Create simulator (combines APF with Dijkstra planning)
simulator = APFSimulator(start, goal, obstacles)

# Run simulation (uses both APF forces and Dijkstra guidance)
path, success = simulator.simulate()

# Visualize results (shows both APF field and Dijkstra path)
simulator.visualize(path)
```

### Advanced Usage with Individual Components

For more control, you can use the individual APF and Dijkstra components:

```python
from artificial_potential_field_example import ArtificialPotentialField, compute_global_path, types

# Define obstacles and goal
obstacles = [(10, 10, 2), (20, 20, 3)]  # (x, y, radius)
goal = (40, 40)
start = (0, 0)

# Create APF instance
apf = ArtificialPotentialField(
    goal, 
    obstacles,
    xi=1.5,      # Attractive gain
    eta=1000.0,  # Repulsive gain
    sigma0=10.0, # Attractive threshold distance
    rho0=5.0,    # Repulsive influence radius
    k_circ=100.0 # Circumferential gain for escaping local minima
)

# Compute global path using Dijkstra
global_path, grid_info = compute_global_path(start, goal, obstacles)

# Compute force at a position
force = apf.compute_total_force((5, 5))
print(f"Force at (5, 5): {force}")
```

### API Reference

- `APFSimulator`: Complete simulation class combining APF with Dijkstra path planning
- `ArtificialPotentialField`: Core APF implementation for local force computation
- `compute_global_path`: Dijkstra-based global path planner on discrete grid
- `generate_random_obstacles`: Utility to generate random obstacles for testing
- `plot_field`: Visualization function for APF fields, Dijkstra paths, and obstacle maps
- `demo_basic_usage`: Interactive demo showing basic APF + Dijkstra features
- `demo_advanced_features`: Demo showing parameter tuning and advanced APF + Dijkstra features
- `demo_comprehensive`: Complete demo covering all APF + Dijkstra functionality
- `run_demo`: Simple demo function that runs the main APF + Dijkstra visualization

See the code and docstrings for more advanced usage and configuration options.
