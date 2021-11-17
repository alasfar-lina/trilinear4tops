import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rc
from scipy import interpolate

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def FancySpiderPlot(ax,data,yticks=[0,1,2,3,4],ytickslabels=None,ylim=(0,6),facecolor=('#EFEFF1','#36394f'),colors=['#006f71','#710100'],DataLabels=['',''],XaxisLabels=None,XaxisPad=-40,gridpad=2.0):
    """
    A function that makes a fancy spider plot with glow effect 
    Input
    -----
    ax : the figure axis, should be set to (polar=True) , recommended plt.subplot(polar=True)
    data: a list of 1 dim np.arrays of the data you wish to plot. should not be circular. 
    yticks: a list consisting of the ticks ( also the labels) for the "y axis"
    ylim:  a tuple (ymin, ymax), if you set y max to a big value you can add decorations
    facecolor: a tuple of the background colour of the plot and the shades of the yaxis circles.
    colors: list of the colours of the lines for each dataset you want to plot
    DataLabels: list of the labels that you want to use in the legend
    XaxisLabels: list of the x-axis labels
    XaxisPad: Padding for the x-axis labels, could be used for styling
    """
   
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")
    ax.xaxis.grid(False)
    ax.set_facecolor(facecolor[0])
    n_lines = 12
    diff_linewidth = 1.05
    alpha_value = 0.03
    l= len(data[0])+1
    angles =np.array([i*360/(l-1) for i in range(l)])
    for degree in angles:
        rad = np.deg2rad(degree)
        ax.plot([rad,rad], [yticks[0],ylim[-1]-gridpad], color="#f3f3f3", linewidth=2., zorder=len(yticks)+1,alpha=0.5)
    for d,col,label in zip(data,colors,DataLabels):
        d =np.array( np.r_[d, d[0]])
        x=d[:-1] * np.cos(angles[:-1]/180*np.pi)
        y=d[:-1] * np.sin(angles[:-1]/180*np.pi)
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
        tck, u = interpolate.splprep([x, y], s=0, per=True)
        # evaluate the spline fits for 1000 evenly spaced distance values
        xi, yi = interpolate.splev(np.linspace(0, 1, 2000), tck)
        #glow effect
        for n in range(1, n_lines+1):
            ax.plot(cart2pol(xi, yi)[1], cart2pol(xi, yi)[0],linewidth=2+(diff_linewidth*n), linestyle='solid',color= col
                    , alpha=alpha_value,zorder= 97)
    # Plot data
        ax.plot(cart2pol(xi, yi)[1], cart2pol(xi, yi)[0],linewidth=2, linestyle='solid',color= col, alpha=1,label=label,zorder= 100)
        ax.plot(angles/180*np.pi, d, linewidth=1, linestyle='',marker='o',color= col,zorder= 99)       
    # Fill area
        ax.fill(cart2pol(xi, yi)[1], cart2pol(xi, yi)[0], col, alpha=0.15, zorder= 98)
    ax.set_ylim(ylim[0],ylim[1])
    fmt = matplotlib.ticker.StrMethodFormatter("{x}")
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xticks(angles[:-1]/180*np.pi)
    ax.set_xticklabels(XaxisLabels)
    if  ytickslabels != None:
        ax.set_yticklabels(ytickslabels)
    ax.tick_params(axis='both', which='major', pad=XaxisPad)
    for i in range(len(yticks)):
        ax.fill_between(
        np.linspace(0, 2*np.pi, 100),  # Need high res or you'll fill a triangle
        yticks[0],
        yticks[i],
        alpha=0.1,
        color=facecolor[1], zorder= i)
    ax.set_yticks(yticks)
    return ax