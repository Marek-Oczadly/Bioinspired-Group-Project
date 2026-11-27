import math
import numpy as np

METRES_PER_KM = 1000
METRES_PER_FOOT = 0.3048
MINUTES_PER_HOUR = 60

# (east, north) offset of each neighbour: (NW, N, NE, W, E, SW, S, SE)
MOORE_OFFSETS = ((-1, 1), (0, 1), (1, 1),
                 (-1, 0), (1, 0),
                 (-1, -1), (0, -1), (1, -1))


def kmh_to_ft_per_min(speed_kmh) -> float:
    return speed_kmh * METRES_PER_KM / METRES_PER_FOOT / MINUTES_PER_HOUR


def ft_per_min_to_kmh(speed_ft_per_min) -> float:
    return (speed_ft_per_min * METRES_PER_FOOT * MINUTES_PER_HOUR
            / METRES_PER_KM)


def neighbour_distances(cell_size, offsets=MOORE_OFFSETS) -> list[float]:
    """Centre-to-centre distance to each neighbour, in the units of cell_size."""
    return [cell_size * math.hypot(east, north) for east, north in offsets]


def wind_components(direction) -> tuple[float, float]:
    """Split a wind direction (degrees, see RunConstants) into east and north
    components, each -1 to 1, for the way the wind is BLOWING TOWARDS."""
    radians = math.radians(direction % 360)
    east = -math.sin(radians)
    north = -math.cos(radians)
    return east, north


def values_at_neighbours(array, offsets=MOORE_OFFSETS) -> list[np.ndarray]:
    """For each offset, an array whose value at a cell is `array` at that
    neighbour (False beyond the grid edge), in the order of offsets."""
    rows, cols = array.shape
    padded = np.pad(array, 1, constant_values=False)
    return [padded[1 - north:1 - north + rows, 1 + east:1 + east + cols]
            for east, north in offsets]
