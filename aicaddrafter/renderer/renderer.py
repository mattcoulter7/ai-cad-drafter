import matplotlib.pyplot as plt
import typing as T
from shapely import LineString, Polygon


def render(
    lines: T.List[LineString],
    polygons:  T.List[Polygon]
):
    plt.figure()

    for line in lines:
        x, y = line.xy  # Extract coordinates from LineString
        plt.plot(x, y)  # Plot the line with markers at each vertex

    for poly in polygons:
        exterior_coords = poly.exterior.coords.xy
        plt.fill(exterior_coords[0], exterior_coords[1], alpha=0.5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()
