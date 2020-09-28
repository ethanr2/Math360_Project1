# -*- coding: utf-8 -*-

from datetime import date
from datetime import datetime as dt
from time import time

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
                "Year", "X_next", "dX_next", "Y_next", "dY_next"
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
                        t + START_YEAR, X_next, dX_next, Y_next, dY_next
                    )
                )
            else:
                print(
                    "{} {:10.2e} {:10.2e} {:10.2e} {:10.2e}".format(
                        t + START_YEAR, X_next, dX_next, Y_next, dY_next
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


def make_matrix(n, X_max=550, Y_max=550):
    step = np.sqrt(X_max*Y_max/n)
    av = (X_max + Y_max)/2
    x_coords = np.arange(0, X_max, step * X_max/av)
    y_coords = np.arange(0, Y_max, step * Y_max/av)
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
            rez = trial(xx[0][i], yy[j][0])
            x_noughts.append(rez["X"][0])
            y_noughts.append(rez["Y"][0])
            dx_noughts.append(rez["dX"][0])
            dy_noughts.append(rez["dY"][0])
            X.append(rez["X"])
            Y.append(rez["Y"])
            ts.append(rez["t"])
            vect_colors.append(np.sqrt(rez["dX"][0] ** 2 + rez["dY"][0] ** 2))
            #print(i, j)
        
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

def vector_field(name,title="Math 360 - Project 1: Vector Field", size=20, proportion=1, n=20, X_max=550, Y_max=550):
    df = make_matrix(n, X_max, Y_max)
    df["X_1"] = df["X_0"] + df["dX_0"] / df["vector_color"] * size *proportion
    df["Y_1"] = df["Y_0"] + df["dY_0"] / df["vector_color"] * size * proportion * Y_max/X_max
    source = ColumnDataSource(df)
    
    p = figure(
        width=800,
        height=800,
        title=title,
        x_axis_label="X",
        y_axis_label="Y",
        x_range=(0, X_max - 50),
        y_range=(0, Y_max - 50),
    )

    mapper = LinearColorMapper(
        palette=Viridis256, low=df["vector_color"].max(), high=df["vector_color"].min()
    )
    colors = {"field": "vector_color", "transform": mapper}

    p.add_layout(
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

    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.background_fill_color = "#4A4A4A"

    return p


def add_path(p, path, proportion=1, shock = False):
    path["X_1"] = path["X"] + path["dX"]
    path["Y_1"] = path["Y"] + path["dY"]
    if shock:
        path["X_1"] = path["X_1"]/10

    p.line(
        x=path.iloc[:-1, 0],
        y=path.iloc[:-1, 1],
        line_color="white",
        line_width=2 / proportion,
    )

    last = path.iloc[-2, :]
    p.add_layout(
        Arrow(
            end=NormalHead(line_color=None, fill_color="white", size=4 / proportion),
            x_start=last["X"],
            y_start=last["Y"],
            x_end=last["X_1"],
            y_end=last["Y_1"],
            line_width=2 / proportion,
            line_color="#FFFFFF",
        )
    )
    return p

def problem_2():
    print("Problem 2")
    p2 = trial(
        200, 300, display=True, num_iter=18, pop_exit=False, shock=False, make_t=False
    )
    p2.to_latex(
        "papers/problem2_tab.tex",
        header=["\(X\)", "\(Y\)", "\(\Delta X\)", "\(\Delta Y\)"],
        escape=False,
        float_format="{:0.3e}".format,
    )
    
def problem_3():
    print("\nProblem 3")
    p3 = trial(
        100, 150, display=True, num_iter=18, pop_exit=False, shock=False, make_t=False
    )

    fig_p3 = vector_field("problem3", title ='Problem 3: Vector Field', proportion=2 / 3, 
                        n=400, X_max=350, Y_max=350)
    
    # Make the white phase diagram
    path1 = p3.loc[:1998, :]
    fig_p3 = add_path(fig_p3, path1, 2 / 3)

    path2 = p3.loc[1998:2008, :]
    fig_p3 = add_path(fig_p3, path2, 2 / 3)

    label_path = p3.loc[[1990, 1998, 2008], :]
    label_path["x_offset"] = [0, 10, 0]
    label_path["y_offset"] = [0, -5, 30]
    label_path = ColumnDataSource(label_path)

    fig_p3.circle(
        x="X", y="Y", size=8, line_color="white", fill_color="White", source=label_path
    )

    lbs = LabelSet(
        x="X",
        y="Y",
        x_offset="x_offset",
        y_offset="y_offset",
        text="Year",
        source=label_path,
        text_font_size="15px",
        text_color="white",
    )
    fig_p3.add_layout(lbs)
    show(fig_p3)
    # Export
    p3.to_latex(
        "papers/problem3_tab.tex",
        header=["\(X\)", "\(Y\)", "\(\Delta X\)", "\(\Delta Y\)"],
        escape=False,
        float_format="{:0.2f}".format,
    )
    export_png(fig_p3, filename="papers/charts/problem3_chart.png")

def problem_4():
    print("\nProblem 4")
    p4 = trial(
        200, 300, display=True, num_iter=18, pop_exit=False, shock=True, make_t=False
    )
    fig_p4 = vector_field("problem4", proportion=2,size = 10, n=1500, X_max=800, Y_max=2000)
    
    # Make the white phase diagram
    path1 = p4.loc[:1995, :]
    fig_p4 = add_path(fig_p4, path1)
    
    path2 = p4.loc[1995:1996, :]
    fig_p4 = add_path(fig_p4, path2, shock = True)
    
    path3 = p4.loc[1996:2000, :]
    fig_p4 = add_path(fig_p4, path3)
    
    label_path = p4.loc[[1990, 1996], :]
    label_path["x_offset"] = [-40, 10]
    label_path["y_offset"] = [0, -5]
    label_path = ColumnDataSource(label_path)
    
    fig_p4.circle(
        x="X", y="Y", size=8, line_color="white", fill_color="White", source=label_path
    )
    
    lbs = LabelSet(
        x="X",
        y="Y",
        x_offset="x_offset",
        y_offset="y_offset",
        text="Year",
        source=label_path,
        text_font_size="15px",
        text_color="white",
    )
    fig_p4.add_layout(lbs)
    
    show(fig_p4)
    
    
    # Export 
    p4.to_latex(
            "papers/problem4_tab.tex",
            header=["\(X\)", "\(Y\)", "\(\Delta X\)", "\(\Delta Y\)"],
            escape=False,
            float_format="{:0.2f}".format,
        )
    export_png(fig_p4, filename="papers/charts/problem4_chart.png")
    

#%%
def main():
    problem_2()
    problem_3()
    problem_4()
main()
#%%
# fig_p3 = vector_field('problem3_img',proportion= 1, n = 35, X_max = 550, Y_max = 550)

# df = make_matrix(500, 500, 500)
# f = make_matrix(25, 500, 500)

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
