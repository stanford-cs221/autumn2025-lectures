from edtrace import link
from typing import Callable, Any
import altair as alt
from altair import Chart, Data
import functools
import numpy as np
import random
import torch


def article_link(url):
    return link(url, title="[article]")


PLOT_SIZE = 300  # Width and height of plots with an arrow, in pixels


def make_plot(title: str | None,
              xlabel: str,
              ylabel: str,
              f: Callable[[float], float] | None,
              xrange: tuple[float, float] = (-3, 3),
              points: list[dict] | None = None,
              line_color: str | None = None,
              above_color: str | None = None,
              below_color: str | None = None,
              arrow: tuple[tuple[float, float], tuple[float, float]] | None = None,
              num_points: int = 30,
              domain: tuple[float, float] | None = None) -> dict:
    """
    If `above_color` or `below_color` is given (requires `points` or `domain`), shade the region above or below the line `f`.
    If `arrow` = (start, end) is given, draw an arrow from start to end (e.g., the weight vector).
    Plots with an arrow use the same scale on both axes, so that angles are drawn faithfully
    (e.g., the weight vector is perpendicular to the decision boundary).
    `num_points` is the number of points at which `f` is sampled (increase for sharp transitions).
    If `domain` is given, use it as the range of both axes.
    """
    to_show = []

    values = [{xlabel: x, ylabel: f(x)} for x in np.linspace(xrange[0], xrange[1], num_points)] if f is not None else []

    # If there are points, pad both axes so that points near the edges are easy to see
    x = alt.X(f"{xlabel}:Q")
    y = alt.Y(f"{ylabel}:Q")
    if domain is not None:
        x_domain, y_domain = list(domain), list(domain)
        x = x.scale(domain=x_domain)
        y = y.scale(domain=y_domain)
        if f is not None:  # Extend the line across the full x-axis (clipped to the y-axis below)
            values = [{xlabel: x, ylabel: f(x)} for x in np.linspace(x_domain[0], x_domain[1], num_points)]
    elif points:
        if arrow is not None:
            # Fit the points and the arrow (the line is clipped), with equal scales on both axes
            domain_points = points + [{xlabel: ax, ylabel: ay} for ax, ay in arrow]
            x_domain = padded_domain([p[xlabel] for p in domain_points])
            y_domain = padded_domain([p[ylabel] for p in domain_points])
            x_domain, y_domain = equalize_spans(x_domain, y_domain)
        else:
            x_domain = padded_domain([p[xlabel] for p in points + values])
            y_domain = padded_domain([p[ylabel] for p in points + values])
        x = x.scale(domain=x_domain)
        y = y.scale(domain=y_domain)
        if f is not None:  # Extend the line across the full x-axis (clipped to the y-axis below)
            values = [{xlabel: x, ylabel: f(x)} for x in np.linspace(x_domain[0], x_domain[1], num_points)]

    if f is not None and (points or domain is not None):
        # Shade regions above/below the line (fill up to the edge of the plot)
        for color, edge in [(above_color, y_domain[1]), (below_color, y_domain[0])]:
            if color is not None:
                region = Chart(Data(values=values)).mark_area(clip=True, color=color, opacity=0.3).encode(x=x, y=y, y2=alt.Y2(datum=edge))
                to_show.append(region)

    if f is not None:
        line = Chart(Data(values=values)).mark_line(clip=True, **({"color": line_color} if line_color else {})).encode(x=x, y=y)
        to_show.append(line)

    if points is not None:
        points = Chart(Data(values=points)).mark_point(filled=True, size=100, opacity=1).encode(x=x, y=y, color=alt.Color("color:N").scale(None))
        to_show.append(points)

    if arrow is not None:
        to_show.extend(make_arrow(arrow, xlabel, ylabel, x, y, color=line_color or "black"))

    chart = functools.reduce(lambda c1, c2: c1 + c2, to_show)
    if arrow is not None or domain is not None:
        chart = chart.properties(width=PLOT_SIZE, height=PLOT_SIZE)
    if title is not None:
        chart = chart.properties(title=title)
    return chart.to_dict()


def make_arrow(arrow: tuple[tuple[float, float], tuple[float, float]], xlabel: str, ylabel: str,
               x: alt.X, y: alt.Y, color: str) -> list[Chart]:
    """Return the charts (shaft and head) for an arrow from `arrow[0]` to `arrow[1]`."""
    (x0, y0), (x1, y1) = arrow
    # Pixels per unit (plot is PLOT_SIZE x PLOT_SIZE pixels), used to orient the head on screen
    x_domain = x.to_dict().get("scale", {}).get("domain")
    y_domain = y.to_dict().get("scale", {}).get("domain")
    x_pixels_per_unit = PLOT_SIZE / (x_domain[1] - x_domain[0]) if x_domain else 1
    y_pixels_per_unit = PLOT_SIZE / (y_domain[1] - y_domain[0]) if y_domain else 1
    dx_pixels, dy_pixels = (x1 - x0) * x_pixels_per_unit, (y1 - y0) * y_pixels_per_unit
    angle = np.degrees(np.arctan2(dx_pixels, dy_pixels)) % 360  # Clockwise from up, in [0, 360)

    # Custom triangle whose tip is at the anchor (0, 0), so the tip lands exactly on the end point
    head_size = 150
    head_shape = "M0,0 L-0.6,1.5 L0.6,1.5 Z"
    head_length_pixels = 1.5 * np.sqrt(head_size) / 2  # Path coordinates are scaled by sqrt(size) / 2
    head = Chart(Data(values=[{xlabel: x1, ylabel: y1}])).mark_point(shape=head_shape, filled=True, size=head_size, opacity=1, color=color, strokeWidth=0, angle=float(angle)).encode(x=x, y=y)

    # Stop the shaft inside the head so its thick end doesn't poke out past the tip
    shrink = max(0, 1 - 0.6 * head_length_pixels / np.hypot(dx_pixels, dy_pixels))
    shaft_end = {xlabel: x0 + (x1 - x0) * shrink, ylabel: y0 + (y1 - y0) * shrink}
    shaft = Chart(Data(values=[{xlabel: x0, ylabel: y0}, shaft_end])).mark_line(color=color, strokeWidth=3).encode(x=x, y=y)
    return [shaft, head]


def equalize_spans(x_domain: list[float], y_domain: list[float]) -> tuple[list[float], list[float]]:
    """Widen the narrower of the two domains (around its center) so both have the same span."""
    span = max(x_domain[1] - x_domain[0], y_domain[1] - y_domain[0])
    def widen(domain: list[float]) -> list[float]:
        center = (domain[0] + domain[1]) / 2
        return [center - span / 2, center + span / 2]
    return widen(x_domain), widen(y_domain)


def padded_domain(values: list[float], fraction: float = 0.15) -> list[float]:
    """Return [min, max] of `values`, extended on both sides by `fraction` of the range (at least 1)."""
    low, high = float(min(values)), float(max(values))
    pad = max((high - low) * fraction, 1)
    return [low - pad, high + pad]


class Vocabulary:
    """Maps strings to integers."""
    def __init__(self):
        self.index_to_string: list[str] = []
        self.string_to_index: dict[str, int] = {}

    def get_index(self, string: str) -> int:  # @inspect string
        index = self.string_to_index.get(string)  # @inspect index
        if index is None:  # New string
            index = len(self.index_to_string)  # @inspect index
            self.index_to_string.append(string)
            self.string_to_index[string] = index
        return index

    def get_string(self, index: int) -> str:
        return self.index_to_string[index]

    def __len__(self):
        return len(self.index_to_string)

    def asdict(self):
        return {
            "index_to_string": self.index_to_string,
            "string_to_index": self.string_to_index,
        }


def set_random_seed(seed: int):
    """Set all random seeds for deterministic behavior."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def one_hot(index: int, length: int) -> torch.Tensor:
    """Create a one-hot vector of the given `length` with a 1 at the `index` position."""
    vector = torch.zeros(length)
    vector[index] = 1
    return vector


def sample_dict(choices: dict[Any, float]) -> Any:
    """Sample a key from a dictionary of choices based on their probabilities (values)."""
    return np.random.choice(list(choices.keys()), p=list(choices.values()))


def normalize_dict(choices: dict[Any, float]) -> dict[Any, float]:
    """Normalize a dictionary of choices based on their probabilities (values)."""
    total_prob = sum(choices.values())
    return {key: prob / total_prob for key, prob in choices.items()}