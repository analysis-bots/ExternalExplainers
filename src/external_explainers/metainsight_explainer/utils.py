import numpy as np
import matplotlib.pyplot as plt

def generate_color_shades(cmap_name: str, num_colors: int, start: float = 0.3, end: float = 0.9) -> list:
    """
    Generates a list of color shades from a specified matplotlib colormap.

    Args:
        cmap_name (str): The name of the colormap (e.g., 'Greens', 'Reds').
        num_colors (int): The number of different shades to generate.
        start (float): The starting point of the colormap (0.0 is lightest).
        end (float): The ending point of the colormap (1.0 is darkest).

    Returns:
        list: A list of RGBA color tuples.
    """
    if num_colors == 0:
        return []
    # Get the colormap object
    cmap = plt.get_cmap(cmap_name)
    # Generate N evenly spaced numbers between start and end
    space = np.linspace(start, end, num_colors)
    # Return the list of colors
    return [cmap(x) for x in space]