# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 03:52:48 2020

@author: ethan
"""

# -*- coding: utf-8 -*-

from datetime import date
from datetime import datetime as dt
from time import time

# import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import lambertw
from scipy.integrate import solve_ivp

from bokeh.io import export_png, output_file, show
from bokeh.plotting import figure
from bokeh.models import (
    LinearColorMapper,
    LabelSet,
    ColumnDataSource,
    Arrow,
    NormalHead,
    OpenHead,
    VeeHead,
)
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column
from bokeh.palettes import Viridis256, Inferno256

pd.set_option("mode.chained_assignment", None)
# Parameters

START_YEAR = 1990
POP_MAX = 10000
a = 0.002
b = .2
c = 0.004
d = .3
ts = []

alpha1 = 1.5
beta1 = 100
alpha2 = 3
beta2 = 200
def model(x, y):
    dX = -a * x**2 + b * x + (alpha1*x*y)/(beta1+y)
    dY = -c * y**2 + d * y  + (alpha2*x*y)/(beta2+x)
    
    return dX, dY
def trial(
    x_0=149,
    y_0=201,
    display=False,
    num_iter=False,
    pop_exit=True,
    shock=False,
    make_t=True,
):
    X = [x_0]
    Y = [y_0]
    dX, dY = model(X[-1], Y[-1])
    dX = [dX]
    dY = [dY]
    iterate = True
    t = 1
    equilibrium = False
    if display:
        print(
            "{:10} {:10} {:10} {:10} {:10}".format(
                "Year", "X", "Y", "dX_next", "dY_next"
            )
        )

    while iterate:
        # print(X[-1],dX )
        X_next = X[-1] + dX[-1]
        Y_next = Y[-1] + dY[-1]
        if shock and t == 6:
            X_next = X_next / 10

        dX_next, dY_next = model(X_next, Y_next)

        X.append(X_next)
        Y.append(Y_next)
        dX.append(dX_next)
        dY.append(dY_next)

        if t > 1:
            equilibrium = (X[-1] == X[-2]) and (Y[-1] == Y[-2])

        if min(X_next, Y_next) < 0.5 or equilibrium:
            color = np.nan
            iterate = False
        if pop_exit and max(X_next, Y_next) > POP_MAX:
            color = t
            iterate = False
        if num_iter and t >= num_iter:
            color = t
            iterate = False
        if display:
            if max(X_next, Y_next) < 1000000:
                print(
                    "{} {:10.2f} {:10.2f} {:10.2f} {:10.2f}".format(
                        t + START_YEAR, X_next, Y_next, dX_next, dY_next
                    )
                )
            else:
                print(
                    "{} {:10.2e} {:10.2e} {:10.2e} {:10.2e}".format(
                        t + START_YEAR, X_next, Y_next, dX_next, dY_next
                    )
                )
        t += 1
    if make_t:
        data = {"X": X, "Y": Y, "dX": dX, "dY": dY, "t": color}
    else:
        data = {"X": X, "Y": Y, "dX": dX, "dY": dY}
        data = pd.DataFrame(data)
        data.index = data.index + START_YEAR
        data.index.name = "Year"
    return data

p3 = trial(
        1, 1, display=True, num_iter=25, pop_exit=False, shock=False, make_t=False
    )
p3



class VectorField(object):
    def __init__(
        self,
        name,
        title="Math 360 - Project 1: Vector Field",
        size=20,
        rescale=True,
        proportion=1,
        n=20,
        X_min=0,
        Y_min=0,
        X_max=550,
        Y_max=550,
    ):
        self.size = size
        self.proportion = proportion

        self.df = make_matrix(n, X_max=X_max, Y_max=Y_max, X_min=X_min, Y_min=Y_min)

        if rescale:
            self.df["X_1"] = (
                self.df["X_0"]
                + self.df["dX_0"] / self.df["vector_color"] * size * proportion
            )
            self.df["Y_1"] = self.df["Y_0"] + self.df["dY_0"] / self.df[
                "vector_color"
            ] * size * proportion * (Y_max - Y_min) / (X_max - X_min)

        else:
            self.df["X_1"] = self.df["X_0"] + self.df["dX_0"] * proportion
            self.df["Y_1"] = self.df["Y_0"] + self.df["dY_0"] * proportion * (
                Y_max - Y_min
            ) / (X_max - X_min)
        source = ColumnDataSource(self.df)

        self.p = figure(
            width=800,
            height=800,
            title=title,
            x_axis_label="X - Bee Population",
            y_axis_label="Y - Clover Population",
            x_range=(X_min, X_max - 5),
            y_range=(Y_min, Y_max - 5),
        )

        mapper = LinearColorMapper(
            palette=Viridis256,
            low=self.df["vector_color"].max(),
            high=self.df["vector_color"].min(),
        )
        colors = {"field": "vector_color", "transform": mapper}

        self.p.add_layout(
            Arrow(
                end=NormalHead(line_color=None, size=2 / proportion),
                x_start="X_0",
                y_start="Y_0",
                x_end="X_1",
                y_end="Y_1",
                line_width=1 / proportion,
                source=source,
                line_color=colors,
            )
        )

        self.p.xaxis[0].ticker.desired_num_ticks = 10
        self.p.xgrid.grid_line_color = None
        self.p.ygrid.grid_line_color = None
        self.p.background_fill_color = "#4A4A4A"

    def add_path(self, path, shock=False, thin=False):
        path["X_1"] = path["X"] + path["dX"]
        path["Y_1"] = path["Y"] + path["dY"]
        if shock:
            path["X_1"] = path["X_1"] / 10

        if thin:
            width = 1 / self.proportion
        else:
            width = 2 / self.proportion

        self.p.line(
            x=path.iloc[:-1, 0],
            y=path.iloc[:-1, 1],
            line_color="white",
            line_width=width,
        )

        last = path.iloc[-2, :]
        self.p.add_layout(
            Arrow(
                end=NormalHead(
                    line_color=None, fill_color="white", size=4 / self.proportion
                ),
                x_start=last["X"],
                y_start=last["Y"],
                x_end=last["X_1"],
                y_end=last["Y_1"],
                line_width=width,
                line_color="#FFFFFF",
            )
        )
def make_matrix(n, X_max=550, Y_max=550, X_min=0, Y_min=0):
    xdiff = X_max - X_min
    ydiff = Y_max - Y_min
    step = np.sqrt(xdiff * ydiff / n)
    av = (xdiff + ydiff) / 2
    x_coords = np.arange(X_min, X_max, step * xdiff / av)
    y_coords = np.arange(Y_min, Y_max, step * ydiff / av)
    xx, yy = np.meshgrid(x_coords, y_coords, sparse=True)

    x_noughts = []
    dx_noughts = []
    X = []
    y_noughts = []
    dy_noughts = []
    Y = []
    ts = []
    step = X_max / n
    arr_i = []
    vect_colors = []
    for i in range(len(xx[0])):
        for j in range(len(yy)):
            rez = trial(xx[0][i], yy[j][0], num_iter = 3, pop_exit = False )
            x_noughts.append(rez["X"][0])
            y_noughts.append(rez["Y"][0])
            dx_noughts.append(rez["dX"][0])
            dy_noughts.append(rez["dY"][0])
            X.append(rez["X"])
            Y.append(rez["Y"])
            ts.append(rez["t"])
            vect_colors.append(np.sqrt(rez["dX"][0] ** 2 + rez["dY"][0] ** 2))
            if n > 3000:
                print(i, j)

    data = {
        "X_0": x_noughts,
        "dX_0": dx_noughts,
        "X": X,
        "Y_0": y_noughts,
        "dY_0": dy_noughts,
        "Y": Y,
        "color": ts,
        "vector_color": vect_colors,
    }
    df = pd.DataFrame(data)

    # Replaces NA values
    df.loc[df["color"].isna(), "color"] = df["color"].max()

    hist = np.histogram(df["color"], density=True, bins=50)
    transform = lambda x: np.searchsorted(hist[1], x)
    df.loc[:, "color"] = df.loc[:, "color"].apply(transform)
    print(df)
    return df
fig =  VectorField(
        "problem7",
        proportion=.75,
        rescale=True,
        size = 30,
        n=1000,
        X_max=1005,
        Y_max=1005,
        title="Problem 7: A Revised Model",
        X_min=0,
        Y_min=0,
    )
p = trial(
        600, 50, display=True, num_iter=18, pop_exit=False, shock=True, make_t=False
    )
#fig.add_path(p)
show(fig.p)
export_png(fig.p, filename="papers/charts/problem7_chart.png")