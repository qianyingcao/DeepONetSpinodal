import os
import torch
import time
import math
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from scipy import io
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import *


def subplot_scatter_color(ax, x, y, z, title, marker='o', mask=None):
    if mask is None:
        cs = ax.scatter(x, y, c=z, marker=marker)
    else:
        x_ = x.flatten()
        y_ = y.flatten()
        z_ = z.flatten()
        mask = mask.flatten()
        with_data = mask>0.5
        without_data = mask<0.5
        cs = ax.scatter(x_[with_data], y_[with_data], c=z_[with_data], marker=marker)
        #cs = ax.scatter(x_, y_, c=z_, marker=marker, alpha=0.3)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    cb = plt.colorbar(cs, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    return

def subplot_contourf(ax, x, y, z, title):
    cs = ax.contourf(x, y, z, 100)
    #cs = ax.contour(x, y, z)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    cb = plt.colorbar(cs, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    return
            

            


