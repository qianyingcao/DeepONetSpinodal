"""
Author: Somdatta Goswami, somdatta_goswami@brown.edu
Plotting xDisplacement, yDisplacement and phi
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')

def plotField(hist_true_print, crackTip, damage_pred_print, damage_true_print, xDisp_pred_print, xDisp_true_print, yDisp_pred_print, yDisp_true_print, istep, folder, segment):
    
        fig = plt.figure(constrained_layout=False, figsize=(12, 12))
        gs = fig.add_gridspec(3, 4)
        plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.3, hspace = 0.2)
        
        ax = fig.add_subplot(gs[1,0])        
        h = ax.imshow(hist_true_print.T,origin='lower', interpolation='nearest', cmap='jet', aspect=1)
        ax.set_title('Initial crack = ' + str(crackTip))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        divider = make_axes_locatable(ax)
        
        ax = fig.add_subplot(gs[0,1])        
        h = ax.imshow(damage_pred_print.T,origin='lower', interpolation='nearest', cmap='jet', aspect=1)
        ax.set_title('Pred $\phi$(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,1])
        h = ax.imshow(damage_true_print.T,origin='lower', interpolation='nearest', cmap='jet', aspect=1)
        ax.set_title('True $\phi$(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,1])
        h = ax.imshow(abs(damage_pred_print.T - damage_true_print.T),origin='lower', interpolation='nearest', cmap='jet', aspect=1)
        ax.set_title('Error in $\phi$(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)        
        h.set_clim(vmin = 0, vmax = 0.2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        x_disp_max = np.max(np.maximum(xDisp_pred_print,xDisp_true_print))
        x_disp_min = np.min(np.minimum(xDisp_pred_print,xDisp_true_print))

        ax = fig.add_subplot(gs[0,2])
        h = ax.imshow(xDisp_pred_print.T,origin='lower', interpolation='nearest', cmap='jet', aspect=1)
        ax.set_title('Pred $u$(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)        
        h.set_clim(vmin=-0.004, vmax=0.012)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,2])
        h = ax.imshow(xDisp_true_print.T,origin='lower', interpolation='nearest', cmap='jet', aspect=1)
        ax.set_title('True $u$(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)        
        h.set_clim(vmin=-0.004, vmax=0.012)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,2])
        h = ax.imshow(abs(xDisp_pred_print.T - xDisp_true_print.T),origin='lower', interpolation='nearest', cmap='jet', aspect=1)
        ax.set_title('Error in $u$(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)        
        h.set_clim(vmin = 0, vmax =0.012)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        y_disp_max = np.max(np.maximum(yDisp_pred_print,yDisp_true_print))
        y_disp_min = np.min(np.minimum(yDisp_pred_print,yDisp_true_print)) 
        
        ax = fig.add_subplot(gs[0,3])
        h = ax.imshow(yDisp_pred_print.T, origin='lower')
        ax.set_title('Pred $v$(x)')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)        
        h.set_clim(vmin=y_disp_min, vmax=y_disp_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,3])
        h = ax.imshow(yDisp_true_print.T,origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('True $v$(x)')
        h.set_clim(vmin=y_disp_min, vmax=y_disp_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,3])
        h = ax.imshow(abs(yDisp_pred_print.T - yDisp_true_print.T),origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('Error in $v$(x)')
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(folder + '/step_' + str(istep) + '.png')
        plt.close()
    
