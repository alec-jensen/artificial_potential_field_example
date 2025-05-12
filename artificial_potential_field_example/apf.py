import math
import random
from .types import Position, Force, Obstacle, Obstacles

class ArtificialPotentialField:
    def __init__(self, goal: Position, obstacles: Obstacles, xi: float = 1.0, eta: float = 100.0, sigma0: float = 5.0, rho0: float = 3.0, k_circ: float = 0.0):
        """
        Initialize the Artificial Potential Field.

        Args:
            goal: Target position (x, y).
            obstacles: List of obstacles (x, y, radius).
            xi: Attractive gain constant (like k_att).
            eta: Repulsive gain constant (like k_rep).
            sigma0: Distance threshold for attractive force behavior change.
            rho0: Radius of influence for repulsive force (obstacles only repel within this distance).
            k_circ: Circumferential gain for rotation field to escape local minima.
        """
        self.goal: Position = goal
        self.obstacles: Obstacles = obstacles
        self.xi: float = xi
        self.eta: float = eta
        self.sigma0: float = sigma0
        self.rho0: float = rho0
        self.k_circ: float = k_circ

    def compute_attractive_force(self, position: Position) -> Force:
        """Compute the attractive force towards the goal based on distance."""
        dx = float(self.goal[0] - position[0])
        dy = float(self.goal[1] - position[1])
        dist_to_goal = math.hypot(dx, dy)

        # Avoid division by zero if at the goal
        if dist_to_goal == 0:
            return (0.0, 0.0)

        if dist_to_goal <= self.sigma0:
            # Fatt(q) = -ξ * (current position - target position)
            # Note: (current - target) is (-dx, -dy)
            fx = self.xi * dx
            fy = self.xi * dy
        else:
            # Fatt(q) = -ξ * σ0 * (current position - target position) / (distance to target)
            # Note: (current - target) / distance = (-dx/dist, -dy/dist)
            fx = self.xi * self.sigma0 * dx / dist_to_goal
            fy = self.xi * self.sigma0 * dy / dist_to_goal

        return (fx, fy)

    def compute_repulsive_force(self, position: Position) -> Force:
        """Compute the repulsive force away from obstacles within the radius of influence."""
        fx_total, fy_total = 0.0, 0.0
        for ox, oy, orad in self.obstacles:
            dx = position[0] - ox
            dy = position[1] - oy
            # Use distance to obstacle center as approximation for ρi(q, qobst)
            dist_to_obstacle = math.hypot(dx, dy)
            
            # Effective distance of influence, considering obstacle radius
            # The repulsive force should start acting *outside* the physical radius
            effective_rho0 = self.rho0 + orad 

            # Check if the point is inside the obstacle - this shouldn't happen ideally,
            # but if it does, apply a strong outward force.
            if dist_to_obstacle <= orad:
                 # Simplified strong push if inside
                 if dist_to_obstacle == 0: # Avoid division by zero if exactly at center
                     # Apply a small random push if at the center
                     rand_angle = random.uniform(0, 2 * math.pi)
                     fx_total += self.eta * math.cos(rand_angle)
                     fy_total += self.eta * math.sin(rand_angle)
                 else:
                     fx_total += self.eta * (dx / dist_to_obstacle) * 10 # Strong push outwards
                     fy_total += self.eta * (dy / dist_to_obstacle) * 10
                 continue # Skip normal calculation if inside

            # Calculate force only if within the influence radius (rho0) but outside the obstacle radius
            if orad < dist_to_obstacle <= effective_rho0:
                # Frepi(q) = η * ( (1 / dist) - (1 / ρ0) ) * (1 / dist)² * grad(dist)
                # grad(dist) = (current pos - obstacle pos) / dist = (dx/dist, dy/dist)
                term1 = (1.0 / dist_to_obstacle) - (1.0 / effective_rho0)
                term2 = 1.0 / (dist_to_obstacle**2)
                
                # Magnitude of the radial repulsive force component
                mag = self.eta * term1 * term2
                
                # Radial repulsion component (points away from obstacle)
                fx_rad = mag * (dx / dist_to_obstacle)
                fy_rad = mag * (dy / dist_to_obstacle)
                
                fx_total += fx_rad
                fy_total += fy_rad

                # Add tangential circulation force if k_circ is non-zero
                # Use the same 'term1' which increases closer to the obstacle
                if self.k_circ != 0:
                    circ_mag = self.k_circ * term1 # Scale circulation by proximity
                    # Tangential component (perpendicular to dx, dy)
                    fx_circ = circ_mag * (-dy / dist_to_obstacle)
                    fy_circ = circ_mag * (dx / dist_to_obstacle)
                    fx_total += fx_circ
                    fy_total += fy_circ
            # else: force is 0 if outside effective_rho0

        return (fx_total, fy_total)

    def compute_total_force(self, position: Position) -> Force:
        """Compute the total force as the sum of attractive and repulsive forces."""
        attractive_force = self.compute_attractive_force(position)
        repulsive_force = self.compute_repulsive_force(position)

        fx = attractive_force[0] + repulsive_force[0]
        fy = attractive_force[1] + repulsive_force[1]
        return (fx, fy)