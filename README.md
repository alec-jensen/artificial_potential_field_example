# Artificial Potential Field Example

This repository demonstrates the implementation of an Artificial Potential Field (APF) algorithm for path planning in a 2D environment with obstacles.

Key features include:

* **Standard APF Implementation:** Uses attractive forces towards the goal and repulsive forces from obstacles based on standard potential field formulas.
* **Configurable Parameters:** Allows tuning of attractive gain (`xi`), repulsive gain (`eta`), attractive threshold (`sigma0`), repulsive influence radius (`rho0`), and circumferential gain (`k_circ`) for escaping local minima.
* **Hybrid Planning:** Combines the local APF with a global path planner (Dijkstra on a grid) to guide the agent and mitigate local minima issues.
* **Momentum:** Incorporates velocity and momentum into the simulation loop for smoother movement.
* **Visualization:** Generates plots showing the force vector field, potential contour map, 3D potential surface, obstacle locations, start/goal points, the global Dijkstra path, and the agent's final path.

![image](https://github.com/user-attachments/assets/4f048e46-8924-4faa-aba0-589f82d56146)
