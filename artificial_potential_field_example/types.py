from typing import Sequence, Tuple, Union

numeric = Union[int, float]

Position = Tuple[numeric, numeric]
Obstacle = Tuple[numeric, numeric, numeric]  # (x, y, radius)
Obstacles = Sequence[Obstacle]
Force = Tuple[numeric, numeric]