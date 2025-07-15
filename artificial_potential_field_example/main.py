#!/usr/bin/env python3

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

def generate_random_obstacles(start: Tuple[float, float], goal: Tuple[float, float], 
                            num_obstacles: int = 15, world_bounds: Tuple[float, float] = (45, 45),
                            min_radius: float = 1.0, max_radius: float = 4.0) -> List[Obstacle]:
    """
    Generate random obstacles that don't intersect with start, goal, or each other.
    
    Args:
        start: Start position (x, y)
        goal: Goal position (x, y)
        num_obstacles: Number of obstacles to generate
        world_bounds: World size (width, height)
        min_radius: Minimum obstacle radius
        max_radius: Maximum obstacle radius
    
    Returns:
        List of obstacles (x, y, radius)
    """
    obstacles: List[Obstacle] = []
    
    for _ in range(num_obstacles):
        attempts = 0
        while attempts < 100:  # Prevent infinite loops
            x = np.random.uniform(0, world_bounds[0])
            y = np.random.uniform(0, world_bounds[1])
            radius = np.random.uniform(min_radius, max_radius)
            
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
            attempts += 1
    
    return obstacles

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

    return path, grid_info
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

class APFSimulator:
    """
    A complete APF simulator that combines APF with Dijkstra path planning.
    """
    
    def __init__(self, start: Tuple[float, float], goal: Tuple[float, float], obstacles: List[Obstacle],
                 xi: float = 1.5, eta: float = 1000.0, sigma0: float = 10.0, 
                 rho0: float = 5.0, k_circ: float = 100.0):
        """
        Initialize the APF simulator.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            obstacles: List of obstacles (x, y, radius)
            xi: Attractive gain constant
            eta: Repulsive gain constant
            sigma0: Distance threshold for attractive force behavior change
            rho0: Radius of influence for repulsive force
            k_circ: Circumferential gain for rotation field
        """
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        
        # Create APF instance
        self.apf = ArtificialPotentialField(goal, obstacles, xi, eta, sigma0, rho0, k_circ)
        
        # Compute global path
        self.global_path, self.grid_info = compute_global_path(start, goal, obstacles)
        
    def simulate(self, max_iterations: int = 1000, step_size: float = 0.5, 
                 beta: float = 0.8, path_guidance_weight: float = 0.7,
                 max_velocity: float = 1.0) -> Tuple[List[Tuple[float, float]], bool]:
        """
        Run the APF simulation.
        
        Args:
            max_iterations: Maximum number of simulation steps
            step_size: Step size for movement
            beta: Momentum coefficient (0=no momentum, 1=full momentum)
            path_guidance_weight: Weight for global path guidance
            max_velocity: Maximum velocity magnitude
            
        Returns:
            Tuple of (path, success) where path is list of positions and success is bool
        """
        position = self.start
        path: List[Tuple[float, float]] = [position]
        
        # Momentum and velocity initialization
        velocity = (0.0, 0.0)
        force_scale = 1.0
        
        # Hybrid global+local: initialize waypoint following
        current_waypoints = list(self.global_path)
        waypoint_idx = 1
        total_waypoints = len(current_waypoints)
        local_threshold = 1.0  # distance to switch waypoint
        
        # Use goal directly if no intermediate waypoints or if path is empty
        if total_waypoints > 1:
            local_goal = current_waypoints[waypoint_idx]
        else:
            local_goal = self.goal
        
        for i in range(max_iterations):
            # Compute force using APF
            force = self.apf.compute_total_force(position)
            
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
            
            # Blend APF force with guidance force (only if following waypoints)
            if current_waypoints:
                blended_force = (
                    (1 - path_guidance_weight) * normalized_force[0] + path_guidance_weight * normalized_guidance[0],
                    (1 - path_guidance_weight) * normalized_force[1] + path_guidance_weight * normalized_guidance[1]
                )
                # Re-normalize the blended force
                blended_mag = (blended_force[0]**2 + blended_force[1]**2)**0.5
                if blended_mag > 0:
                    normalized_force = (blended_force[0]/blended_mag, blended_force[1]/blended_mag)
                else:
                    normalized_force = (0, 0)
            
            # Update velocity with momentum
            velocity = (
                beta * velocity[0] + (1 - beta) * normalized_force[0] * step_size * force_scale,
                beta * velocity[1] + (1 - beta) * normalized_force[1] * step_size * force_scale
            )
            
            # Clamp velocity to maximum magnitude
            vel_mag = (velocity[0]**2 + velocity[1]**2)**0.5
            if vel_mag > max_velocity:
                velocity = (velocity[0]/vel_mag * max_velocity, velocity[1]/vel_mag * max_velocity)
            
            # Update position using velocity
            position = (position[0] + velocity[0], position[1] + velocity[1])
            path.append(position)
            
            # Check if reached local waypoint (only if following waypoints)
            if current_waypoints:
                dist_loc = ((position[0] - local_goal[0])**2 + (position[1] - local_goal[1])**2)**0.5
                if dist_loc < local_threshold:
                    waypoint_idx += 1
                    if waypoint_idx >= total_waypoints:
                        local_goal = self.goal
                        current_waypoints = []
                    else:
                        local_goal = current_waypoints[waypoint_idx]
            
            # Check if reached final goal
            dist_final_goal = ((position[0] - self.goal[0])**2 + (position[1] - self.goal[1])**2)**0.5
            if dist_final_goal < ROBOT_RADIUS:
                return path, True
        
        return path, False
    
    def visualize(self, path: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        Visualize the APF field and path.
        
        Args:
            path: Optional path to visualize. If None, runs simulation first.
        """
        if path is None:
            path, _ = self.simulate()
        
        plot_field(self.apf, self.start, self.goal, self.obstacles, path, self.global_path, self.grid_info)

def main():
    """Run a simple demonstration of the artificial potential field."""
    # Demo parameters
    start = (0, 0)
    goal = (40, 40)
    
    # Generate random obstacles
    obstacles = generate_random_obstacles(start, goal)
    
    # Create simulator
    simulator = APFSimulator(start, goal, obstacles)
    
    # Run simulation
    path, success = simulator.simulate()
    
    print(f"Starting at position: {start}")
    print(f"Goal position: {goal}")
    print(f"Using robot radius: {ROBOT_RADIUS} with safety margin: {SAFETY_MARGIN}")
    
    if success:
        print(f"Reached goal in {len(path)} steps.")
    else:
        print("Failed to reach goal within maximum iterations.")
    
    # Visualize results
    simulator.visualize(path)


def run_demo():
    """Run the APF demo as a function, for modular use."""
    main()

def demo_basic_usage():
    """Demonstrate basic usage of the APF module components."""
    print("=== APF Module Demo - Basic Usage ===")
    
    # Setup
    start = (5, 5)
    goal = (35, 35)
    obstacles: List[Obstacle] = [(15, 15, 3), (25, 10, 2), (20, 25, 4)]
    
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Obstacles: {obstacles}")
    print()
    
    # 1. Basic APF usage
    print("1. Creating APF instance and computing forces:")
    apf = ArtificialPotentialField(goal, obstacles, xi=2.0, eta=500.0)
    
    test_positions = [(0, 0), (10, 10), (20, 20), (30, 30)]
    for pos in test_positions:
        force = apf.compute_total_force(pos)
        attractive = apf.compute_attractive_force(pos)
        repulsive = apf.compute_repulsive_force(pos)
        print(f"  Position {pos}: Total={force}, Attractive={attractive}, Repulsive={repulsive}")
    print()
    
    # 2. Global path planning
    print("2. Computing global path with Dijkstra:")
    global_path, grid_info = compute_global_path(start, goal, obstacles)
    print(f"  Global path has {len(global_path)} waypoints")
    if global_path:
        print(f"  First few waypoints: {global_path[:3]}")
    print()
    
    # 3. Complete simulation
    print("3. Running complete APF simulation:")
    simulator = APFSimulator(start, goal, obstacles)
    path, success = simulator.simulate(max_iterations=500)
    
    if success:
        print(f"  ✓ Successfully reached goal in {len(path)} steps")
        print(f"  Final position: {path[-1]}")
    else:
        print(f"  ✗ Failed to reach goal after {len(path)} steps")
    print()
    
    # 4. Visualization (optional)
    print("4. Visualization available via simulator.visualize(path)")
    print("   Call simulator.visualize() to see the complete field visualization")
    print()
    
    return simulator, path, success

def demo_advanced_features():
    """Demonstrate advanced features like parameter tuning and obstacle generation."""
    print("=== APF Module Demo - Advanced Features ===")
    
    # 1. Random obstacle generation
    print("1. Generating random obstacles:")
    start = (2, 2)
    goal = (38, 38)
    obstacles = generate_random_obstacles(
        start, goal, 
        num_obstacles=10, 
        world_bounds=(40, 40),
        min_radius=1.0, 
        max_radius=3.0
    )
    print(f"  Generated {len(obstacles)} random obstacles")
    print(f"  Sample obstacles: {obstacles[:3]}")
    print()
    
    # 2. Parameter comparison
    print("2. Comparing different APF parameters:")
    test_pos = (20, 20)
    
    # Different parameter sets
    param_sets = [
        {"xi": 1.0, "eta": 100.0, "name": "Low repulsion"},
        {"xi": 2.0, "eta": 1000.0, "name": "High repulsion"},
        {"xi": 0.5, "eta": 500.0, "name": "Weak attraction"},
    ]
    
    for params in param_sets:
        apf = ArtificialPotentialField(
            goal, obstacles, 
            xi=params["xi"], 
            eta=params["eta"]
        )
        force = apf.compute_total_force(test_pos)
        print(f"  {params['name']}: Force at {test_pos} = {force}")
    print()
    
    # 3. Simulation with different settings
    print("3. Running simulations with different momentum settings:")
    simulator = APFSimulator(start, goal, obstacles)
    
    momentum_settings = [
        {"beta": 0.0, "name": "No momentum"},
        {"beta": 0.5, "name": "Medium momentum"},
        {"beta": 0.9, "name": "High momentum"},
    ]
    
    for setting in momentum_settings:
        path, success = simulator.simulate(
            max_iterations=300,
            beta=setting["beta"],
            step_size=0.3
        )
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {setting['name']}: {status}, {len(path)} steps")
    print()
    
    return simulator

def demo_comprehensive():
    """Run a comprehensive demo showing all features."""
    print("=== Comprehensive APF Module Demo ===")
    print()
    
    # Run basic demo
    simulator1, path1, success1 = demo_basic_usage()
    
    # Run advanced demo
    simulator2 = demo_advanced_features()
    
    print("=== Summary ===")
    print("This demo showcased:")
    print("- Basic APF force computation")
    print("- Global path planning with Dijkstra")
    print("- Complete APF simulation")
    print("- Random obstacle generation")
    print("- Parameter tuning effects")
    print("- Momentum settings comparison")
    print()
    print("For visualization, call:")
    print("  simulator.visualize(path)")
    print("or:")
    print("  simulator.visualize()  # runs simulation first")
    print()
    
    return simulator1, simulator2

if __name__ == "__main__":
    main()