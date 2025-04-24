from .types import Position, Force, Obstacle, Obstacles
import random
import numpy as np

class ArtificialPotentialField:
    def __init__(self, goal: Position, obstacles: Obstacles):
        self.goal: Position = goal
        self.obstacles: Obstacles = obstacles
        self.k_att: float = 1.0  # Attractive gain
        self.k_rep: float = 100.0  # Increased repulsive gain
        self.influence_factor: float = 1.5  # Increased influence radius multiplier
        self.noise_factor: float = 0.01  # Small random noise to break symmetry

    def compute_attractive_force(self, position: Position) -> Force:
        # Compute the attractive force towards the goal
        dx = float(self.goal[0] - position[0])
        dy = float(self.goal[1] - position[1])
        
        # Simple linear attractive force
        return (dx, dy)

    def compute_repulsive_force(self, position: Position) -> Force:
        # Compute the repulsive force from obstacles
        repulsive_force_x: float = 0.0
        repulsive_force_y: float = 0.0
        for obstacle in self.obstacles:
            dx: float = position[0] - obstacle[0]
            dy: float = position[1] - obstacle[1]
            distance = np.sqrt(dx**2 + dy**2)
            radius: float = obstacle[2]
            influence_radius: float = radius * self.influence_factor

            # inside of the obstacle is max repulsion, with a little push to wherever is the closest
            # side of the obstacle
            if distance < radius:
                # Repulsive force is maximum when inside the obstacle
                repulsive_force_x += (1.0 / max(distance, 0.1)) * dx
                repulsive_force_y += (1.0 / max(distance, 0.1)) * dy
                # Add small random noise to break symmetry
                repulsive_force_x += random.uniform(-self.noise_factor, self.noise_factor)
                repulsive_force_y += random.uniform(-self.noise_factor, self.noise_factor)
            elif distance < influence_radius:
                # Repulsive force is inversely proportional to the distance
                repulsive_force_x += (1.0 / max(distance, 0.1)) * dx
                repulsive_force_y += (1.0 / max(distance, 0.1)) * dy
                # Add small random noise to break symmetry
                repulsive_force_x += random.uniform(-self.noise_factor, self.noise_factor)
                repulsive_force_y += random.uniform(-self.noise_factor, self.noise_factor)
            else:
                # Outside of the influence radius, no repulsive force
                continue

        return (repulsive_force_x, repulsive_force_y)

    def compute_total_force(self, position: Position) -> Force:
        attractive_force: Force = self.compute_attractive_force(position)
        repulsive_force: Force = self.compute_repulsive_force(position)
        
        total_force_x: float = self.k_att * attractive_force[0] + self.k_rep * repulsive_force[0]
        total_force_y: float = self.k_att * attractive_force[1] + self.k_rep * repulsive_force[1]
        return (total_force_x, total_force_y)