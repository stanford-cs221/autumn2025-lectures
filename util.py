from edtrace import link
from typing import Callable
from altair import Chart, Data
import functools
import numpy as np


def article_link(url):
    return link(url, title="[article]")


def make_plot(title: str,
              xlabel: str,
              ylabel: str,
              f: Callable[[float], float] | None,
              xrange: tuple[float, float] = (-3, 3),
              points: list[dict] | None = None) -> dict:
    to_show = []

    if f is not None:
        values = [{xlabel: x, ylabel: f(x)} for x in np.linspace(xrange[0], xrange[1], 30)]
        line = Chart(Data(values=values)).properties(title=title).mark_line().encode(x=f"{xlabel}:Q", y=f"{ylabel}:Q")
        to_show.append(line)

    if points is not None:
        points = Chart(Data(values=points)).mark_point().encode(x=f"{xlabel}:Q", y=f"{ylabel}:Q", color="color:N")
        to_show.append(points)

    chart = functools.reduce(lambda c1, c2: c1 + c2, to_show)
    return chart.to_dict()

