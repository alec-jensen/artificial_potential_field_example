#!/usr/bin/env python3

# Use absolute imports when running as script
if __name__ == "__main__":
    from artificial_potential_field_example import ArtificialPotentialField
    from artificial_potential_field_example.types import Obstacle
else:
    # Use relative imports when imported as a module
    from . import ArtificialPotentialField
    from .types import Obstacle

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.patches as patches  # For drawing circles
from typing import cast
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import heapq  # For global Dijkstra planner

obstacles = []

goal = (40, 40)  # Goal position
start = (0, 0)  # Start position

# create n random obstacles that dont intersect the start, goal, or each other
for _ in range(25):
    while True:
        x = np.random.uniform(0, 45)
        y = np.random.uniform(0, 45)
        radius = np.random.uniform(1, 4)
        # Check if the obstacle intersects with start or goal
        if (x - start[0])**2 + (y - start[1])**2 > (radius + 1)**2 and \
           (x - goal[0])**2 + (y - goal[1])**2 > (radius + 1)**2:
            # Check if the obstacle intersects with existing obstacles
            intersects = False
            for obs in obstacles:
                if (x - obs[0])**2 + (y - obs[1])**2 < (radius + obs[2])**2:
                    intersects = True
                    break
            if not intersects:
                obstacles.append((x, y, radius))
                break

# Instantiate APF with the new parameters
apf = ArtificialPotentialField(
    goal,
    obstacles,
    xi=1.0,      # Attractive gain (formerly k_att)
    eta=500.0,   # Repulsive gain (formerly k_rep)
    sigma0=10.0, # Attractive threshold distance
    rho0=5.0,    # Repulsive influence radius (beyond obstacle radius)
    k_circ=100.0 # Circumferential gain (kept)
)

# Define global path computation using Dijkstra on a discrete grid

def compute_global_path(start: Tuple[float, float], goal: Tuple[float, float], obstacles: List[Obstacle],
                         grid_min=0, grid_max=45, grid_step=1, robot_radius=1.0) -> List[Tuple[float, float]]:
    # Build grid coordinates
    xs = np.arange(grid_min, grid_max + grid_step, grid_step)
    ys = np.arange(grid_min, grid_max + grid_step, grid_step)
    nx, ny = len(xs), len(ys)
    # Occupancy grid: True if free
    free = np.ones((nx, ny), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for ox, oy, orad in obstacles:
                if np.hypot(x - ox, y - oy) <= orad + robot_radius:
                    free[i, j] = False
                    break
    # Find grid indices for start/goal
    start_idx = (np.argmin(np.abs(xs - start[0])), np.argmin(np.abs(ys - start[1])))
    goal_idx = (np.argmin(np.abs(xs - goal[0])), np.argmin(np.abs(ys - goal[1])))
    # Dijkstra with 8-connected neighbors
    dist = {start_idx: 0.0}
    prev = {}
    hq = [(0.0, start_idx)]
    offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while hq:
        d, u = heapq.heappop(hq)
        if u == goal_idx:
            break
        if d > dist[u]:
            continue
        ux, uy = u
        for dx, dy in offsets:
            vx, vy = ux + dx, uy + dy
            if 0 <= vx < nx and 0 <= vy < ny and free[vx, vy]:
                nd = d + np.hypot(dx, dy)
                v = (vx, vy)
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(hq, (nd, v))
    # Reconstruct path
    path = []
    node = goal_idx
    while node in prev:
        path.append((xs[node[0]], ys[node[1]]))
        node = prev[node]
    path.append((xs[start_idx[0]], ys[start_idx[1]]))
    path.reverse()
    return path

# Compute global path waypoints
global_path = compute_global_path(start, goal, obstacles)
print(f"Computed global path with {len(global_path)} waypoints.")
print(f"Global path waypoints: {global_path}")

def calculate_potential(apf: ArtificialPotentialField, x: float, y: float) -> float:
    """Calculate the potential field value at a given point (x, y)."""
    # Attractive potential (distance to goal)
    dx = x - apf.goal[0]
    dy = y - apf.goal[1]
    attractive_potential = np.sqrt(dx**2 + dy**2)
    
    # Repulsive potential (inverse distance to obstacles)
    repulsive_potential = 0
    for obs in apf.obstacles:
        dx = x - obs[0]
        dy = y - obs[1]
        distance = np.sqrt(dx**2 + dy**2)
        radius = obs[2]
        if distance < radius * 2:  # Influence radius is twice the physical radius
            repulsive_potential += (1.0 / max(distance, 0.1)) * 10  # Scale for visibility
    
    # Total potential
    return attractive_potential + repulsive_potential

def plot_field(apf: ArtificialPotentialField, start: Tuple, goal: Tuple, obstacles: List, path: Optional[List[Tuple]] = None, global_path: Optional[List[Tuple]] = None) -> None:
    """
    Visualize the artificial potential field, obstacles, goal, and path.
    
    Args:
        apf: The artificial potential field object
        start: Starting position (x, y)
        goal: Goal position (x, y)
        obstacles: List of obstacles (x, y, radius)
        path: List of positions along the path
        global_path: List of positions along the global path
    """
    # Ensure path and global_path lists and initialize coordinate arrays
    if path is None:
        path = []
    if global_path is None:
        global_path = []
    path_x: List[float] = []
    path_y: List[float] = []
    global_x: List[float] = []
    global_y: List[float] = []
    # Create subplots: add a 3rd subplot for 3D surface
    fig = plt.figure(figsize=(22, 7))
    ax1 = fig.add_subplot(1, 3, 1)  # Vector field
    ax2 = fig.add_subplot(1, 3, 2)  # Contour plot of potential field
    ax3 = cast(Axes3D, fig.add_subplot(1, 3, 3, projection='3d'))  # 3D surface

    # Create a grid of points
    x = np.linspace(-5, 45, 40)
    y = np.linspace(-5, 45, 40)
    X, Y = np.meshgrid(x, y)

    # Calculate forces and potential at each point
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    Z = np.zeros_like(X)  # For potential field

    for i in range(len(x)):
        for j in range(len(y)):
            pos = (X[j, i], Y[j, i])
            force = apf.compute_total_force(pos)
            magnitude = np.sqrt(force[0]**2 + force[1]**2)
            if magnitude > 0:
                U[j, i] = force[0] / magnitude
                V[j, i] = force[1] / magnitude
            else:
                U[j, i], V[j, i] = 0, 0
            Z[j, i] = calculate_potential(apf, pos[0], pos[1])

    # Plot vector field
    ax1.quiver(X, Y, U, V, color='lightblue', width=0.002)
    for obs in obstacles:
        circle = patches.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5)
        ax1.add_patch(circle)
    ax1.plot(start[0], start[1], 'bo', markersize=10, label='Start')
    ax1.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
    # Plot actual APF path first
    if path:
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        ax1.plot(path_x, path_y, 'g-', linewidth=2, label='Path', zorder=1)  # Changed 'k.-' to 'g-'
    # Plot global Dijkstra path on top with transparency
    if global_path:
        global_x = [pos[0] for pos in global_path]
        global_y = [pos[1] for pos in global_path]
        ax1.plot(global_x, global_y, 'r--', linewidth=2, label='Global Path', alpha=0.6, zorder=2)
    ax1.set_xlim(-5, 45)
    ax1.set_ylim(-5, 45)
    ax1.set_aspect('equal')
    ax1.set_title('Artificial Potential Field - Force Vectors')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)

    # Plot potential field as a contour plot
    contour = ax2.contourf(X, Y, Z, 20, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='Potential Value')
    for obs in obstacles:
        circle = patches.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5)
        ax2.add_patch(circle)
    ax2.plot(start[0], start[1], 'bo', markersize=10, label='Start')
    ax2.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
    # Plot actual APF path first
    if path:
        ax2.plot(path_x, path_y, 'g-', linewidth=2, label='Path', zorder=1)  # Changed 'k.-' to 'g-'
    # Plot global Dijkstra path on top with transparency
    if global_path:
        ax2.plot(global_x, global_y, 'r--', linewidth=2, label='Global Path', alpha=0.6, zorder=2)
    ax2.set_xlim(-5, 45)
    ax2.set_ylim(-5, 45)
    ax2.set_aspect('equal')
    ax2.set_title('Artificial Potential Field - Potential Values')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)

    # --- 3D Surface Plot ---
    surf = ax3.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax3.set_title('Artificial Potential Field - 3D Surface')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Potential')
    # Plot obstacles as cylinders (approximate with circles at z=0)
    for obs in obstacles:
        phi = np.linspace(0, 2 * np.pi, 50)
        cx = obs[0] + obs[2] * np.cos(phi)
        cy = obs[1] + obs[2] * np.sin(phi)
        cz = np.zeros_like(cx)
        ax3.plot(cx, cy, cz, color='red', alpha=0.7)
    # Plot start and goal
    ax3.scatter(start[0], start[1], int(calculate_potential(apf, start[0], start[1])), c='b', s=50, label='Start')
    ax3.scatter(goal[0], goal[1], int(calculate_potential(apf, goal[0], goal[1])), c='g', s=80, marker='*', label='Goal')
    # Plot actual APF path first
    if path:
        path_z = [calculate_potential(apf, px, py) for px, py in zip(path_x, path_y)]
        ax3.plot(path_x, path_y, path_z, 'g-', linewidth=2, label='Path', zorder=1)  # Changed 'k.-' to 'g-'
    # Plot global Dijkstra path on top with transparency
    if global_path:
        global_z = [calculate_potential(apf, x, y) for x, y in zip(global_x, global_y)]
        ax3.plot(global_x, global_y, global_z, 'r--', linewidth=2, label='Global Path', alpha=0.6, zorder=2)
    ax3.legend()
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=10, label='Potential Value')

    plt.tight_layout()
    plt.savefig('apf_visualization.png')
    plt.show()

def main():
    """Run a simple demonstration of the artificial potential field."""
    position = start
    print(f"Starting at position: {position}")
    print(f"Goal position: {goal}")
    print(f"Obstacles: {obstacles}")
    
    # Keep track of the path
    path: List[Tuple[float, float]] = [position]
    
    # --- Momentum and velocity initialization ---
    velocity = (0.0, 0.0)
    beta = 0.8  # Momentum coefficient (0=no momentum, 1=full momentum)
    force_scale = 1.0  # Optionally scale the force contribution
    path_guidance_weight = 0.7  # Increased guidance weight
    max_velocity = 1.0  # Maximum velocity magnitude

    # Run simulation until goal is reached or maximum iterations
    max_iterations = 1000
    step_size = 0.5  # Step size for movement
    
    # Hybrid global+local: initialize waypoint following
    waypoint_idx = 1
    total_waypoints = len(global_path)
    local_threshold = 1.0  # distance to switch waypoint
    # Use goal directly if no intermediate waypoints
    if total_waypoints > 1:
        local_goal = global_path[waypoint_idx]
    else:
        local_goal = goal
    
    i = 0
    for i in range(max_iterations):
        # Compute force using APF towards local goal
        force = apf.compute_total_force(position)
        
        # Normalize force for more consistent step sizes
        force_magnitude = (force[0]**2 + force[1]**2)**0.5
        if force_magnitude > 0:
            normalized_force = (force[0]/force_magnitude, force[1]/force_magnitude)
        else:
            normalized_force = (0, 0)
        
        # Compute guidance direction toward current local waypoint
        guidance_vec = (local_goal[0] - position[0], local_goal[1] - position[1])
        guidance_mag = (guidance_vec[0]**2 + guidance_vec[1]**2)**0.5
        if guidance_mag > 0:
            normalized_guidance = (guidance_vec[0]/guidance_mag, guidance_vec[1]/guidance_mag)
        else:
            normalized_guidance = (0.0, 0.0)
        # Blend APF force with guidance force
        blended_force = (
            (1 - path_guidance_weight) * normalized_force[0] + path_guidance_weight * normalized_guidance[0],
            (1 - path_guidance_weight) * normalized_force[1] + path_guidance_weight * normalized_guidance[1]
        )
        # Override normalized_force to include guidance
        normalized_force = blended_force
        
        # --- Update velocity with momentum ---
        velocity = (
            beta * velocity[0] + (1 - beta) * normalized_force[0] * step_size * force_scale,
            beta * velocity[1] + (1 - beta) * normalized_force[1] * step_size * force_scale
        )
        # Clamp velocity to maximum magnitude
        vel_mag = (velocity[0]**2 + velocity[1]**2)**0.5
        if vel_mag > max_velocity:
            velocity = (velocity[0]/vel_mag * max_velocity, velocity[1]/vel_mag * max_velocity)
        
        # --- Update position using velocity ---
        new_position = (position[0] + velocity[0], position[1] + velocity[1])
        
        if i % 10 == 0 or i < 10:  # Print only every 10th step after the first 10
            print(f"Step {i+1}")
            print(f"  Position: {position}")
            print(f"  Force: {force}")
            print(f"  Velocity: {velocity}")
            print(f"  New Position: {new_position}")
        
        position = new_position
        path.append(position)
        
        # Check if reached local waypoint
        dist_loc = ((position[0] - local_goal[0])**2 + (position[1] - local_goal[1])**2)**0.5
        if dist_loc < local_threshold:
            waypoint_idx += 1
            if waypoint_idx >= total_waypoints:
                print(f"Reached final waypoint. Goal achieved.")
                break
            local_goal = global_path[waypoint_idx]
            print(f"Switching to next waypoint: {local_goal}")
    
    if i == max_iterations - 1:
        print(f"Maximum iterations ({max_iterations}) reached without finding goal.")
    
    # Visualize the field and path
    plot_field(apf, start, goal, obstacles, path, global_path)

if __name__ == "__main__":
    main()