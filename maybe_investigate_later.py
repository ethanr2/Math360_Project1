# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 20:27:41 2020

@author: ethan
"""

def f(x, y): return (-c*y+d*x*y)/(-a*x+d*x*y)

def impr_euler(x_0, x_f, y_0, h = 1):
    X = np.arange(x_0, x_f, h)
    Y = [y_0]
    for x in X[1:]:
        k_1 = f(x,Y[-1])
        u = Y[-1] + h*k_1
        k_2 = f(x+h, u)
        Y.append(Y[-1] + h*(k_1 + k_2)/2)
    data = {"X": X, "Y": Y}
    return pd.DataFrame(data)


def make_tbl(X,Y):
    dX, dY = model(X,Y)       
    data = {
        "X_0": X,
        "dX_0": dX,
        "Y_0": Y,
        "dY_0": dY,
        "vector_color": np.sqrt(dX ** 2 + dY ** 2),
    }

    data = pd.DataFrame(data)
    return data

def vector_field(name,title="Math 360 - Project 1: Vector Field" size=20, proportion=1, n=20, X_max=550, Y_max=550, IVP = False):
    if IVP:
        x = np.linspace(149, 10, 100)
        # y = lambertw(-np.exp(-x)*x, k = 0)
        sol_l = solve_ivp(f,method = 'Radau', t_span=[150, 9], y0=[200], t_eval =x )
        x = np.linspace(151, 500, 100)
        sol_r = solve_ivp(f, method = 'Radau', t_span=[150, 500], y0=[200], t_eval =x )
        X = np.append(sol_l.t, sol_r.t)
        Y = np.append(sol_l.y,sol_r.y)
        df= make_tbl(X, Y)
    else:
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

#show(vector_field('test', size=20, proportion=1, n=20, X_max=550, Y_max=550, IVP = True))

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
