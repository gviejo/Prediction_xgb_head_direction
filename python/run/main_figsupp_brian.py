#!/usr/bin/env python

'''
	File name: main_fig4.py
	Author: Guillaume Viejo
	Date created: 30/03/2017    
	Python Version: 2.7

	PLOT figure 4 for cross-corr and timing

'''

import warnings
import pandas as pd
import scipy.io
import numpy as np
# Should not import fonctions if already using tensorflow for something else
import sys, os
import itertools
import cPickle as pickle
from sklearn.model_selection import KFold


#####################################################################
# PLOTTING
#####################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale   # width in inches
	fig_height = fig_width*golden_mean             # height in inches
	# fig_height = 4.696
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)

def noaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)	
	# ax.get_xaxis().tick_bottom()
	# ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height], polar = True)    
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize*2.0)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def get_rgb(mapH, mapS, mapV):
	from matplotlib.colors import hsv_to_rgb	
	"""
		1. convert mapH to x between -1 and 1
		2. get y value between 0 and 1 -> mapV
		3. rescale mapH between 0 and 0.6
		4. normalize mapS

	"""		
	# x = mapH.copy() * 2.0
	# x = x - 1.0
	# y = 1.0 - 0.4*x**6.0
	# mapV = y.copy()
	H   = mapH
	S 	= mapS	
	V 	= mapV
	# HSV = np.dstack((H,S,V))	
	RGB = hsv_to_rgb([H,S,V])	
	return RGB

import matplotlib as mpl

mpl.use("pdf")



pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	"text.usetex": True,                # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 7,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 4,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 6,
	"ytick.labelsize": 6,
	"figure.figsize": figsize(1),     # default fig size of 0.9 textwidth
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.5,
	"ytick.major.size"      : 1.5,
	"xtick.major.size"      : 1.5
	}    
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *

n_pos_brian = 10
n_adn_brian = 10


fig = figure(figsize = figsize(1))


matplotlib.rcParams.update({"axes.labelsize": 	6,
							"font.size": 		8,
							"legend.fontsize": 	8,
							"xtick.labelsize": 	5,
							"ytick.labelsize": 	5,   
							})               # Make the legend/label fonts a little smaller


gs1 	= gridspec.GridSpec(n_pos_brian+1, n_adn_brian+1)
gs1.update(hspace = 0.2,wspace = 0.1, bottom = 0.0, top = 0.97, right = 1.0, left = 0.0)

store_cc 	= pd.HDFStore("../data/fig6_corr_simu_examples_10n.h5")
corr_simu 	= store_cc['data'] 
tcurves 	= store_cc['tcurves']
store_cc.close()

# corr_simu = corr_simu.replace(np.inf, np.nan)
# corr_simu = corr_simu.dropna(axis = 1, how = 'any')
z1 = corr_simu

phi = np.vstack(tcurves.filter(regex='ADn.*').idxmax().values) - tcurves.filter(regex='Pos.*').idxmax().values
phi += 2*np.pi
phi %= 2*np.pi
rgb = phi / (2*np.pi)

from matplotlib.colors import hsv_to_rgb	

for i in range(n_adn_brian):
	ax = subplot(gs1[i+1,0], projection = 'polar')
	# ax.plot(tcurves['ADn.'+str(i)])
	y = tcurves['ADn.'+str(i)]	
	ax.fill_between(y.index.values, np.zeros(len(y)), y, color = hsv_to_rgb([y.idxmax()/(2*np.pi), 1, 1]))
	ax.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
	ax.set_yticks([])
	ax.set_xticklabels([])
	# ax.set_ylim(0,tcurves.filter(regex="ADn.*").max().max())
	if i == 0:
		ax.set_title("ADn")

for j in range(n_pos_brian):
	ax = subplot(gs1[0,j+1], projection = 'polar')	
	y = tcurves['Pos.'+str(j)]	
	ax.fill_between(y.index.values, np.zeros(len(y)), y, color = hsv_to_rgb([y.idxmax()/(2*np.pi), 1, 1]))
	ax.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
	ax.set_yticks([])
	ax.set_xticklabels([])
	# ax.set_ylim(0,tcurves.filter(regex="Pos.*").max().max())
	if j == 0:
		ax.set_ylabel("PoSub", fontsize = 12)

for i in range(n_adn_brian):
	for j in range(n_pos_brian):
		ax = subplot(gs1[i+1,j+1])
		y = corr_simu[i*n_pos_brian + j]
		fill_between(y.index.values, np.zeros(len(y)), y.values, color = 'black')
		# noaxis(ax)
		xlim(-500,500)
		ax.set_xticks([])
		ax.set_yticks([])		
		ax.patch.set_facecolor(hsv_to_rgb([rgb[i,j], 1, 1]))

savefig("../../figures/fig_supp_XCbrian.pdf", dpi=900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig_supp_XCbrian.pdf &")


