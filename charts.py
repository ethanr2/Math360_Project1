
# -*- coding: utf-8 -*-

from datetime import date
from datetime import datetime as dt
from time import time

import numpy as np
import pandas as pd
from scipy import stats

from bokeh.io import export_png,output_file,show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, LabelSet,ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column
from bokeh.palettes import Viridis256
# Parameters

START_YEAR = 1990
POP_MAX = 10000
a = 0.2
b = 0.001
c = 0.3
d = .002
ts = []

def trial(x_0 = 149, y_0 = 201, num_iter = False, display = False):
    X = [x_0]
    Y = [y_0]
    iterate = True
    t = 0
    equilibrium = False
    while iterate:
        X_next = X[-1] - a*X[-1] + b*X[-1]*Y[-1]
        Y_next = Y[-1] - c*Y[-1] + d*X[-1]*Y[-1]
        
        X.append(X_next)
        Y.append(Y_next)
        #print(t+START_YEAR, X_next, Y_next)
        if t > 1:
            equilibrium = (X[-1] == X[-2]) and (Y[-1] == Y[-2])   
        if min(X_next, Y_next) < 0.5 or equilibrium:
            color = np.nan 
            iterate = False 
        if max(X_next, Y_next) > POP_MAX:
            color = t
            iterate = False
        if not num_iter and t >= num_iter:
            color = t
            iterate = False
        t += 1
    return {
        'X': X,
        'Y': Y,
        't': color
        }
    
def make_matrix(n, X_max =300, Y_max = 300):
    x_noughts = []
    X = []
    y_noughts = []
    Y = []
    ts = []
    step = X_max/n
    arr_i= []
    for i in range(1,n):
        arr_j= []
        for j in range(1,n):
            arr_j.append((i*step,j*step))
            rez = trial(i*step,j*step)
            x_noughts.append(rez['X'][0])
            y_noughts.append(rez['Y'][0])
            X.append(rez['X'])
            Y.append(rez['Y'])
            ts.append(rez['t'])
            print(rez['t'])
        arr_i.append(arr_j)
    df = pd.DataFrame({
        'X_0': x_noughts,
        'X': X,
        'Y_0': y_noughts,
        'Y': Y,
        'color': ts
        })
    
    # Replaces NA values
    df.loc[df['color'].isna(), 'color'] = df['color'].max()
    
    hist = np.histogram(df['color'], density = True, bins = 50)
    transform = lambda x: np.searchsorted(hist[1], x)
    df.loc[:, 'color'] =df.loc[:, 'color'].apply(transform) 
    print(df)
    return df



df = make_matrix(20, 300, 300)
source = ColumnDataSource(df)

#%%
#trial(200,300)
p = figure(width = 1000, height = 1000,
           title="Math 360 Project 1" , 
           x_axis_label = 'X', 
           y_axis_label = 'Y',
           y_range = (0,500), x_range=(0,500)
           #x_axis_type = 'log', y_axis_type = 'log'
           )

#p.line(xrng,[0,0], color = 'black')
#p.line([0,0],yrng, color = 'black')
mapper = LinearColorMapper( palette=Viridis256, low=df['color'].max(), high=df['color'].min())
colors= { 'field': 'color', 'transform': mapper}

# row = trial(100,200)
# xdata = row['X']
# ydata = row['Y']
# p.line(xdata, ydata, color = 'black')
df= make_matrix2(25, 500,500)
for i,row in df.iterrows():
    xdata = row['X_0']
    ydata = row['Y_0']
    p.line(xdata, ydata, color = 'black')
#p.circle(x='X_0',y= 'Y_0', color = colors, source = source, size = 2)
p.xaxis[0].ticker.desired_num_ticks = 10
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
#p.xaxis.formatter=NumeralTickFormatter(format="0.0%")
#p.yaxis.formatter=NumeralTickFormatter(format="0.0%")

name = str(dt.now())[:19].replace(' ','').replace(':','-')
export_png(p, filename = 'imgs/{}.png'.format(name))
show(p)

#%%

df
#%%
def set_up(x, y, truncated = True, margins = None):
    if truncated: 
        b = (3 * y.min() - y.max())/2
    else:
        b = y.min()
    if margins == None:    
        xrng = (x.min(),x.max())
        yrng = (b,y.max())
    else:
        xrng = (x.min() - margins,x.max() + margins)
        yrng = (b - margins,y.max() + margins)
        
    x = x.dropna()
    y = y.dropna()
    
    return(x,y,xrng,yrng)

# Chart of non-stationary time series, e.g. NGDP from 2008 to 2020    
def chart0(df):
    xdata, ydata, xrng, yrng = set_up(df.index,df['___'])
    
    p = figure(width = 1000, height = 500,
               title= '____', 
               x_axis_label = 'Date', x_axis_type = 'datetime',
               y_axis_label = '', 
               y_range = yrng, x_range = xrng)
    p.line(xrng,[0,0], color = 'black')
    
    p.line(xdata,ydata, color = 'blue', legend = '')
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.legend.location = 'top_left'
    p.ygrid.grid_line_color = None
    p.yaxis.formatter=NumeralTickFormatter(format="____")
    
    export_png(p,filename ='imgs/chart0.png')

    return p


# Chart of approximately stionary time series, e.g. PCE-Core inflation from 2008 to 2020
def chart1(df):
    xdata, ydata, xrng, yrng = set_up(df.index, df['__'], truncated = False)
    
    p = figure(width = 1000, height = 500,
               title="_" , 
               x_axis_label = 'Date', x_axis_type = 'datetime',
               y_axis_label = '_', 
               y_range = yrng, x_range = xrng)
    p.line(xrng,[0,0], color = 'black')
    
    p.line(xdata,ydata, color = 'blue', legend = '_')
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.legend.location = 'bottom_right'
    p.ygrid.grid_line_color = None
    p.yaxis.formatter=NumeralTickFormatter(format="0.0%")

    export_png(p, filename='imgs/chart1.png')

    return p

# Chart of a regression e.g. inflation vs money supply
def chart2(df):
    xdata, ydata, xrng, yrng = set_up(df['_'], df['_'], 
                                      truncated = False, margins = .005)
    
    p = figure(width = 500, height = 500,
               title="_" , 
               x_axis_label = '_', 
               y_axis_label = '_', 
               y_range = yrng, x_range = xrng)
    p.line(xrng,[0,0], color = 'black')
    p.line([0,0],yrng, color = 'black')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
    leg = 'R = {:.4f}, P-Value = {:.4e}, Slope = {:.4f}'.format(r_value,p_value,slope)
    p.line(xdata, xdata*slope + intercept, legend = leg, color = 'black')
    p.circle(xdata,ydata, color = 'blue',size = 2)
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.formatter=NumeralTickFormatter(format="0.0%")
    p.yaxis.formatter=NumeralTickFormatter(format="0.0%")
    
    export_png(p, filename='images/chart2.png' )
    
    return p
