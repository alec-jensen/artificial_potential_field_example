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

# Safety constants
ROBOT_RADIUS = 1.0  # Physical size of the robot
SAFETY_MARGIN = 0.5  # Extra buffer distance beyond robot radius
TOTAL_SAFE_RADIUS = ROBOT_RADIUS + SAFETY_MARGIN  # Total safe distance from obstacles

obstacles: list[Obstacle] = []

goal = (40, 40)  # Goal position
start = (0, 0)  # Start position

# create n random obstacles that dont intersect the start, goal, or each other
for _ in range(15):
    while True:
        x = np.random.uniform(0, 45)
        y = np.random.uniform(0, 45)
        radius = np.random.uniform(1, 4)
        # Check if the obstacle intersects with start or goal
        if (x - start[0])**2 + (y - start[1])**2 > (radius + TOTAL_SAFE_RADIUS)**2 and \
           (x - goal[0])**2 + (y - goal[1])**2 > (radius + TOTAL_SAFE_RADIUS)**2:
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
    xi=1.5,      # Attractive gain (formerly k_att)
    eta=1000.0,   # Repulsive gain (formerly k_rep)
    sigma0=10.0, # Attractive threshold distance
    rho0=5.0,    # Repulsive influence radius (beyond obstacle radius)
    k_circ=100.0 # Circumferential gain (kept)
)

# Define global path computation using Dijkstra on a discrete grid

# TODO: multiple resolution levels for the grid? i.e. use a finer grid closer to the robot and a coarser grid further away
def compute_global_path(start: Tuple[float, float], goal: Tuple[float, float], obstacles: List[Obstacle],
                         grid_min=0, grid_max=45, grid_step=0.5, robot_radius=None) -> Tuple[List[Tuple[float, float]], dict]:
    # Use the defined total safe radius if not specified
    if robot_radius is None:
        robot_radius = TOTAL_SAFE_RADIUS
    
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
    goal_reached = False
    while hq:
        d, u = heapq.heappop(hq)
        if u == goal_idx:
            goal_reached = True
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
    if goal_reached:
        node = goal_idx
        while node in prev:
            path.append((xs[node[0]], ys[node[1]]))
            node = prev[node]
        path.append((xs[start_idx[0]], ys[start_idx[1]]))
        path.reverse()
    else:
        print("Warning: Global path planner (Dijkstra) could not find a path to the goal.")
    
    # Return path and grid information
    grid_info = {
        'free': free,
        'xs': xs,
        'ys': ys,
        'start_idx': start_idx,
        'goal_idx': goal_idx,
        'dist': dist
    }
    return path, grid_info

# Compute global path waypoints
# Rename the module-level variable to avoid confusion
global_path_computed, grid_info = compute_global_path(start, goal, obstacles)
if not global_path_computed:
    print("Warning: Global path is empty. APF will navigate directly towards the goal without global guidance.")
else:
    print(f"Computed global path with {len(global_path_computed)} waypoints.")

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

def plot_field(apf: ArtificialPotentialField, start: Tuple, goal: Tuple, obstacles: List, path: Optional[List[Tuple]] = None, global_path: Optional[List[Tuple]] = None, grid_info: Optional[dict] = None) -> None:
    """
    Visualize the artificial potential field, obstacles, goal, and path.
    
    Args:
        apf: The artificial potential field object
        start: Starting position (x, y)
        goal: Goal position (x, y)
        obstacles: List of obstacles (x, y, radius)
        path: List of positions along the path
        global_path: List of positions along the global path
        grid_info: Dictionary containing occupancy grid information
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
    
    # Determine figure layout based on whether we have grid_info
    fig = plt.figure(figsize=(22, 14))  # Increase height to accommodate 2 rows
    
    if grid_info:
        # Create 2x2 grid with 4 subplots
        ax1 = fig.add_subplot(2, 2, 1)  # Vector field
        ax2 = fig.add_subplot(2, 2, 2)  # Contour plot of potential field
        ax3 = cast(Axes3D, fig.add_subplot(2, 2, 3, projection='3d'))  # 3D surface
        ax4 = fig.add_subplot(2, 2, 4)  # Occupancy grid
    else:
        # Original layout without occupancy grid in a 1x3 grid
        fig = plt.figure(figsize=(22, 7))
        ax1 = fig.add_subplot(1, 3, 1)  # Vector field
        ax2 = fig.add_subplot(1, 3, 2)  # Contour plot of potential field
        ax3 = cast(Axes3D, fig.add_subplot(1, 3, 3, projection='3d'))  # 3D surface
        ax4 = None  # Explicitly set to None when not used

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
    
    # Plot occupancy grid if available
    if grid_info and ax4 is not None:
        # Extract grid data
        free = grid_info['free']
        xs = grid_info['xs']
        ys = grid_info['ys']
        start_idx = grid_info['start_idx'] 
        goal_idx = grid_info['goal_idx']
        
        # Create a distance-based coloring for the occupancy grid
        # Initialize with high values
        distance_map = np.ones_like(free, dtype=float) * float('inf')
        
        # Fill in distances where available in the dist dictionary
        for (i, j), dist_val in grid_info['dist'].items():
            if 0 <= i < len(xs) and 0 <= j < len(ys):
                distance_map[i, j] = dist_val
        
        # Create a masked array for visualization (mask where free=False)
        masked_distance = np.ma.masked_where(~free, distance_map)
        
        # Occupancy grid visualization - use tuple instead of list for extent
        ax4.imshow(~free.T, origin='lower', extent=(xs[0], xs[-1], ys[0], ys[-1]), 
                  cmap='Greys', alpha=0.7, aspect='auto')
        
        # Plot distances as a heatmap overlay on free cells
        if goal_idx in grid_info['dist']:  # Only if path exists
            dist_plot = ax4.imshow(masked_distance.T, origin='lower', 
                                  extent=(xs[0], xs[-1], ys[0], ys[-1]),
                                  cmap='plasma_r', alpha=0.5, aspect='auto')
            plt.colorbar(dist_plot, ax=ax4, label='Distance from start')
        
        # Plot start and goal
        ax4.plot(xs[start_idx[0]], ys[start_idx[1]], 'bo', markersize=10, label='Start')
        ax4.plot(xs[goal_idx[0]], ys[goal_idx[1]], 'g*', markersize=15, label='Goal')
        
        # Plot actual APF path
        if path:
            ax4.plot(path_x, path_y, 'g-', linewidth=2, label='Path', zorder=1)
            
        # Plot global Dijkstra path
        if global_path:
            ax4.plot(global_x, global_y, 'r--', linewidth=2, label='Global Path', alpha=0.8, zorder=2)
        
        # Add obstacle markers
        for obs in obstacles:
            circle = patches.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5)
            ax4.add_patch(circle)
            
        ax4.set_title('Occupancy Grid with Path Planning')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.legend()
        ax4.grid(True)
        
    plt.tight_layout()
    plt.savefig('apf_visualization.png')
    plt.show()

def main():
    """Run a simple demonstration of the artificial potential field."""
    position = start
    print(f"Starting at position: {position}")
    print(f"Goal position: {goal}")
    print(f"Using robot radius: {ROBOT_RADIUS} with safety margin: {SAFETY_MARGIN}")
    
    # Keep track of the path
    path: List[Tuple[float, float]] = [position]
    
    # --- Momentum and velocity initialization ---
    velocity = (0.0, 0.0)
    beta = 0.8  # Momentum coefficient (0=no momentum, 1=full momentum)
    force_scale = 1.0  # Optionally scale the force contribution
    path_guidance_weight = 0.7  # Increased guidance weight (was 0.4)
    max_velocity = 1.0  # Maximum velocity magnitude

    # Run simulation until goal is reached or maximum iterations
    max_iterations = 1000
    step_size = 0.5  # Step size for movement
    
    # --- Hybrid global+local: initialize waypoint following ---
    # Use a local variable to manage the waypoints being followed. Make a copy.
    current_waypoints = list(global_path_computed)
    waypoint_idx = 1
    total_waypoints = len(current_waypoints)
    local_threshold = 1.0  # distance to switch waypoint
    # Use goal directly if no intermediate waypoints or if path is empty
    if total_waypoints > 1:
        local_goal = current_waypoints[waypoint_idx]
    else:
        local_goal = goal
    
    i = 0
    for i in range(max_iterations):
        # Compute force using APF (implicitly uses final goal)
        force = apf.compute_total_force(position)
        
        # Normalize force for more consistent step sizes
        force_magnitude = (force[0]**2 + force[1]**2)**0.5
        if force_magnitude > 0:
            normalized_force = (force[0]/force_magnitude, force[1]/force_magnitude)
        else:
            normalized_force = (0, 0)
        
        # Compute guidance direction toward current local waypoint
        current_local_goal = local_goal # Use the dynamic local_goal
        guidance_vec = (current_local_goal[0] - position[0], current_local_goal[1] - position[1])
        guidance_mag = (guidance_vec[0]**2 + guidance_vec[1]**2)**0.5
        if guidance_mag > 0:
            normalized_guidance = (guidance_vec[0]/guidance_mag, guidance_vec[1]/guidance_mag)
        else:
            normalized_guidance = (0.0, 0.0)
        
        # Blend APF force with guidance force (only if following waypoints)
        if current_waypoints: # Check the local list of waypoints
            blended_force = (
                (1 - path_guidance_weight) * normalized_force[0] + path_guidance_weight * normalized_guidance[0],
                (1 - path_guidance_weight) * normalized_force[1] + path_guidance_weight * normalized_guidance[1]
            )
            # Re-normalize the blended force
            blended_mag = (blended_force[0]**2 + blended_force[1]**2)**0.5
            if blended_mag > 0:
                 normalized_force = (blended_force[0]/blended_mag, blended_force[1]/blended_mag)
            else:
                 normalized_force = (0,0) # Reset if magnitude is zero
            # If not current_waypoints, normalized_force remains the pure APF force
        
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
        
        position = new_position
        path.append(position)
        
        # Check if reached local waypoint (only if following waypoints)
        if current_waypoints: # Check the local list
            dist_loc = ((position[0] - local_goal[0])**2 + (position[1] - local_goal[1])**2)**0.5
            if dist_loc < local_threshold:
                waypoint_idx += 1
                if waypoint_idx >= total_waypoints:
                    print(f"Reached final waypoint. Switching to final goal.")
                    local_goal = goal
                    current_waypoints = [] # Clear the local list to stop following
                else:
                    local_goal = current_waypoints[waypoint_idx]
                    print(f"Switching to next waypoint ({waypoint_idx}/{total_waypoints-1}): {local_goal}")
        
        # Check if reached final goal - use ROBOT_RADIUS as the threshold
        dist_final_goal = ((position[0] - goal[0])**2 + (position[1] - goal[1])**2)**0.5
        if dist_final_goal < ROBOT_RADIUS:
             print(f"Reached final goal at step {i+1}.")
             break
    
    if i == max_iterations - 1:
        print(f"Maximum iterations ({max_iterations}) reached without finding goal.")
    
    # Visualize the field and path, passing the original computed path and grid info for visualization
    plot_field(apf, start, goal, obstacles, path, global_path_computed, grid_info)

if __name__ == "__main__":
    main()