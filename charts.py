# -*- coding: utf-8 -*-

from datetime import date
from datetime import datetime as dt
from time import time

import numpy as np
import pandas as pd
from scipy import stats

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

# Parameters

START_YEAR = 1990
POP_MAX = 10000
a = 0.2
b = 0.001
c = 0.3
d = 0.002
ts = []


def model(x, y):
    dX = -a * x + b * x * y
    dY = -c * y + d * x * y
 
    return (dX, dY)


def trial(x_0=149, y_0=201, display=False, num_iter=False, pop_exit=True, shock = False, make_t = True):
    X = [x_0]
    Y = [y_0]
    dX, dY = model(X[-1], Y[-1])
    dX = [dX]
    dY = [dY]
    iterate = True
    t = 1
    equilibrium = False
    while iterate:
        # print(X[-1],dX )
        X_next = X[-1] + dX[-1]
        Y_next = Y[-1] + dY[-1]
        if shock and t == 6:
            X_next= X_next/10

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
                print("{} {:10.2f} {:10.2f} {:10.2f} {:10.2f}".format(t + START_YEAR, X_next,dX_next, Y_next, dY_next))
            else:
                print("{} {:10.2e} {:10.2e} {:10.2e} {:10.2e}".format(t + START_YEAR, X_next,dX_next, Y_next, dY_next))
        t += 1
    if make_t:
        data = {"X":X, "Y": Y,"dX": dX, "dY": dY, "t": color}
    else:
        data =  {"X":X, "Y": Y,"dX": dX, "dY": dY}
        data = pd.DataFrame(data)
        data.index = data.index+START_YEAR
        data.index.name = 'Year'
    return data

def make_matrix(n, X_max=300, Y_max=300):
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
    for i in range(1, n):
        arr_j = []
        for j in range(1, n):
            arr_j.append((i * step, j * step))
            rez = trial(i * step, j * step)
            x_noughts.append(rez["X"][0])
            y_noughts.append(rez["Y"][0])
            dx_noughts.append(rez["dX"][0])
            dy_noughts.append(rez["dY"][0])
            X.append(rez["X"])
            Y.append(rez["Y"])
            ts.append(rez["t"])
            vect_colors.append(np.sqrt(rez["dX"][0] ** 2 + rez["dY"][0] ** 2))
            print(i, j)
        arr_i.append(arr_j)
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





def histogram_coloring():
    source = ColumnDataSource(df)
    p = figure(
        width=800,
        height=800,
        title="Histogram Coloring of Initial Points",
        x_axis_label="X",
        y_axis_label="Y",
        y_range=(0, 500),
        x_range=(0, 500),
    )

    mapper = LinearColorMapper(
        palette=Viridis256, low=df["color"].max(), high=df["color"].min()
    )
    colors = {"field": "color", "transform": mapper}

    p.circle(x="X_0", y="Y_0", color=colors, source=source, size=1.5)
    p.xaxis[0].ticker.desired_num_ticks = 10

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    name = str(dt.now())[:19].replace(" ", "").replace(":", "-")
    export_png(p, filename="imgs/histo_coloring{}.png".format(name))

    return p

def histogram_coloring2():
    source = ColumnDataSource(df)
    p = figure(
        width=800,
        height=800,
        title="Histogram Coloring of Initial Points",
        x_axis_label="dX",
        y_axis_label="dY",
       # y_range=(0, 500),
       # x_range=(0, 500),
    )

    mapper = LinearColorMapper(
        palette=Viridis256, low=df["color"].max(), high=df["color"].min()
    )
    colors = {"field": "color", "transform": mapper}

    p.circle(x="dX_0", y="dY_0", color=colors, source=source, size=1.5)
    p.xaxis[0].ticker.desired_num_ticks = 10

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    name = str(dt.now())[:19].replace(" ", "").replace(":", "-")
    export_png(p, filename="imgs/histo_coloringdxdxy{}.png".format(name))

    return p
# df = make_matrix(250, -500, -500)
# show(histogram_coloring2())

def phase_space(x_0=100, y_0=200):
    p = figure(
        width=800,
        height=800,
        title="Phase Space: X = {}, Y = {}".format(x_0, y_0),
        x_axis_label="X",
        y_axis_label="Y",
        y_range=(0, 300),
        x_range=(0, 300),
    )

    mapper = LinearColorMapper(
        palette=Viridis256, low=df["color"].max(), high=df["color"].min()
    )
    colors = {"field": "color", "transform": mapper}

    row = trial(x_0, y_0)
    xdata = row["X"]
    ydata = row["Y"]
    p.line(xdata, ydata, color="black")
    # for i, row in df.iterrows():
    #     xdata = row["X_0"]
    #     ydata = row["Y_0"]
    #     p.line(xdata, ydata, color="black")

    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    name = str(dt.now())[:19].replace(" ", "").replace(":", "-")
    export_png(p, filename="imgs/phase_space{}.png".format(name))

    return p


def vector_field(size=20):
    df["X_1"] = df["X_0"] + df["dX_0"] / df["vector_color"] * size
    df["Y_1"] = df["Y_0"] + df["dY_0"] / df["vector_color"] * size
    source = ColumnDataSource(df)
    p = figure(
        width=800,
        height=800,
        title="Vector Field",
        x_axis_label="X",
        y_axis_label="Y",
        y_range=(0, 500),
        x_range=(0, 500),
    )

    mapper = LinearColorMapper(
        palette=Inferno256, low=df["vector_color"].max(), high=df["vector_color"].min()
    )
    colors = {"field": "vector_color", "transform": mapper}

    # for i, row in df.iterrows():
    #     x = row["X_0"]
    #     y = row["Y_0"]
    #     dx = row['dX_0']
    #     dy = row['dY_0']
    p.add_layout(Arrow(
            end=NormalHead(fill_color = 'yellow',size=2),
            x_start="X_0",
            y_start="Y_0",
            x_end="X_1",
            y_end="Y_1",
            source=source,
            line_color=colors,
      )  
    )

    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    name = str(dt.now())[:19].replace(" ", "").replace(":", "-")
    export_png(p, filename="imgs/vector_field{}.png".format(name))

    return p


# Problem 1
p1 = trial(200,300, display = True, num_iter = 18, pop_exit = False, shock = False, make_t = False)
p1.to_latex('papers/problem1_tab.tex', 
            header = ['\(X\)', '\(Y\)', '\(\Delta X\)', '\(\Delta Y\)'],
            escape = False,
            float_format ="{:0.3e}".format )
#px = trial(200,150, display = True, num_iter = 18, pop_exit = False, shock = False)

# df = make_matrix(500, 500, 500)
#f = make_matrix(25, 500, 500)

# #df = make_matrix(30, 500, 500)
# name = str(dt.now())[:19].replace(" ", "").replace(":", "-")
# output_file("imgs/vector_field{}.html".format(name))
# show(vector_field())
#%%

# import holoviews as hv
# hv.extension('bokeh')
# size = 7
# df = make_matrix(100, 500, 500)
# mag =df["vector_color"]*size
# angle = (np.pi/2.) - np.arctan2(df['dX_0']/mag, df['dY_0']/mag)
# #df["Y_1"] = df["Y_0"] + df["dY_0"] #/ df["vector_color"] * size
# data = (df['X_0'], df['Y_0'], angle, mag)
# vf = hv.VectorField(data)
# vf.opts(color='Magnitude',
#         magnitude = 'Magnitude',
#         width=800,
#         height=800,
#         title="Vector Field",
#         xlabel="X",
#         ylabel="Y",
#         #y_range=(0, 500),
#         #x_range=(0, 500)
#         )
# fig = hv.render(img)
# name = str(dt.now())[:19].replace(" ", "").replace(":", "-")
# output_file("imgs/vector_field{}.html".format(name))
# show(fig)
# #name = str(dt.now())[:19].replace(" ", "").replace(":", "-")
# #hv.save(vf, "imgs/vector_field{}.png".format(name))
# vf
# #%%