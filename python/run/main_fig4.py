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
import xgboost as xgb

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################

#####################################################################
# TIME SPLIT LOADING
#####################################################################
# time_data = pickle.load(open("../data/fig3_timesplit.pickle", 'rb'))
time_data = {}
for ep in ['wake', 'rem', 'sws']:
	time_data[ep] = pickle.load(open("../data/fig4_time_peer_"+ep+"_guillaume.pickle"))


#####################################################################
# CROSS CORR LOADING
#####################################################################
cross_corr_data = pickle.load(open("../data/fig3_cross_correlation.pickle", 'rb'))

times = np.arange(-500, 505, 5)

corr_data = {}

for e,i in zip(['wake', 'rem', 'sleep'], range(3)):	
	corr_data[e] = []
	for s in cross_corr_data.keys():
		for n in cross_corr_data[s][e].keys():
			corr_data[e].append(cross_corr_data[s][e][n])			

	corr_data[e] = np.vstack(corr_data[e])

corr_data['sws'] = corr_data['sleep']




#####################################################################
# PLOTTING
#####################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean *1.             # height in inches
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

def myticks(x,pos):
	if x == 0: return "$0$"
	exponent = int(np.log10(x))
	coeff = x/10**exponent
	return r"${:2.0f} \times 10^{{ {:2d} }}$".format(coeff,exponent)

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

methods = ['xgb_run']

labels = {'mb_10':'MB \n 10 bins', 
			'mb_60':'MB \n 60 bins', 
			'mb_360':'MB \n 360 bins', 
			'lin_comb':'Lin', 
			'nn':'NN', 
			'xgb_run':'XGB'}

colors_ = {'ADn':'#EE6C4D', 'Pos':'#3D5A80'}


labels_plot = [labels[m] for m in methods[0:-1]]


x = np.arange(41)*25 - 20*25
ind = (x>-450)&(x<450)
# ind = np.arange(len(x))
xt = x[ind]



figure(figsize = figsize(1))
outer = gridspec.GridSpec(2,4, wspace = 0.5, hspace = 0.3, width_ratios =[1.,1.,1.,0.5])

matplotlib.rcParams.update({"axes.labelsize": 	6,
							"font.size": 		8,
							"legend.fontsize": 	8,
							"xtick.labelsize": 	5,
							"ytick.labelsize": 	5,   
							})               # Make the legend/label fonts a little smaller
# outer = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec = outerspace[1], hspace = 0.3)

titles = ['Wake', 'REM sleep', 'Slow wave sleep']

for e,i in zip(['wake', 'rem', 'sleep'], range(3)):
	subplot(outer[0,i])
	simpleaxis(gca())
	tmp = []
	for s in cross_corr_data.keys():
		for n in cross_corr_data[s][e].keys():
			tmp.append(cross_corr_data[s][e][n])
			# plot(times, cross_corr_data[s][e][n], '-', color = 'grey', alpha = 0.3)

	title(titles[i], y = 1.04)
	tmp = np.vstack(tmp)
	tmp = tmp[~np.isnan(tmp.mean(1))]
	index = np.logical_and(times > -450, times < 450)
	tmp = tmp[:,index]
	plot(times[index], tmp.mean(0), color = 'green')
	fill_between(times[index], tmp.mean(0)-tmp.var(0), tmp.mean(0)+tmp.var(0), color = 'green', alpha = 0.2, linewidth = 0.0)
	ylabel("Cross-Correlation (ADn / PoSub)")
	axvline(0, color = 'black', linewidth = 0.8)
	xlabel("Time (ms)")
	bbox = gca().get_position().bounds
	ai = axes([bbox[0]+bbox[2]*0.7,bbox[1]+bbox[3]*0.8, 0.06, 0.05])
	simpleaxis(ai)
	# ai.spines['bottom'].set_visible(False)
	ai.spines['left'].set_visible(False)
	ai.set_xticks([], [])
	ai.set_yticks([], [])
	ai.set_title("$\sigma ^2$", fontsize = 4, y = 0.8)
	ai.plot(times[index], tmp.var(0), color = 'green', alpha = 1)
	ai.axvline(0, color = 'black')

###############################################
def normalize(array):
	array -= np.vstack(array.min(1))
	array /= np.vstack(array.max(1))
	return array




subplot(outer[1,0])
simpleaxis(gca())		
ep = 'wake'
# for k in xrange(len(time_data[ep])):
# 	plot(xt, time_data[ep][k][ind], color = colors_['ADn'], alpha = 0.1, linewidth = 0.5)
#mean
plot([0], color = 'none', label = 'Wake')
plot(xt, time_data[ep].mean(0)[ind], color = colors_['ADn'], alpha = 1, linewidth = 2)
ylabel("Gain (a.u.) (ADn $\Rightarrow$ PoSub)", labelpad = 8)
xlabel("Time (ms)")
# title("ADn $\Rightarrow$ PoSub")
# legend(frameon = False)
axvline(0, color = 'black', linewidth = 0.8)
# xlim(-425, 425)
xticks([-400,-200,0,200,400],[-400,-200,0,200,400])


subplot(outer[1,1])
simpleaxis(gca())		
ep = 'rem'
# for k in xrange(len(time_data[ep])):
# 	plot(xt, time_data[ep][k][ind], color = colors_['ADn'], alpha = 0.1, linewidth = 0.5)
#mean
plot([0], color = 'none', label = 'REM sleep')
plot(xt,time_data[ep].mean(0)[ind], color = colors_['ADn'], linewidth = 2)
ylabel("Gain (a.u.) (ADn $\Rightarrow$ PoSub)", labelpad = 8)
xlabel("Time (ms)")
# legend(frameon = False)
axvline(0, color = 'black', linewidth = 0.8)
# xlim(-425, 425)
xticks([-400,-200,0,200,400],[-400,-200,0,200,400])


subplot(outer[1,2])
simpleaxis(gca())		
ep = 'sws'
for k in xrange(len(time_data[ep])):
	if np.max(time_data[ep][k]) < 80.0:
		plot(xt, time_data[ep][k][ind], color = colors_['ADn'], alpha = 0.1, linewidth = 0.5)
#mean
plot([0], color = 'none', label = 'Slow wave sleep')
plot(xt, time_data[ep].mean(0)[ind], color = colors_['ADn'], linewidth = 2)
ylabel("Gain (a.u.) (ADn $\Rightarrow$ PoSub)", labelpad = 8)
xlabel("Time (ms)")
# xlim(-425, 425)
# legend(frameon = False)
axvline(0, color = 'black', linewidth = 0.8)
xticks([-400,-200,0,200,400],[-400,-200,0,200,400])

# ####################################################


# subplot(outer[1,3])
# simpleaxis(gca())		
# for ep in ['wake', 'rem', 'sws']:
	
# 	time_data[ep] = normalize(time_data[ep])
# 	# corr_data[ep] = normalize(corr_data[ep])
	
# 	c1, bins1 = np.histogram(time_data[ep].sum(1), 10)
# 	# c2, bins2 = np.histogram(corr_data[ep].sum(1)/5.)
	
# 	plot(bins1[0:-1], c1/float(c1.sum())*100.0, label = 'xgboost', color = colors_['ADn'])
# 	# plot(bins2[0:-1], c2/float(c2.sum())*100.0, label = 'correlation', color = 'green')
	
# xlabel('Measure precision (a.u.)')
# ylabel('$\%$')

# subplot(outer[0,3])
# simpleaxis(gca())		
# for ep in ['wake', 'rem', 'sws']:
# 	# time_data[ep] = normalize(time_data[ep])

# 	corr_data[ep] = corr_data[ep][~np.isnan(corr_data[ep].mean(1))]

# 	corr_data[ep] = normalize(corr_data[ep])

# 	# c1, bins1 = np.histogram(time_data[ep].sum(1))
# 	c2, bins2 = np.histogram(corr_data[ep].sum(1)/5., 100)

# 	# plot(bins1[0:-1], c1/float(c1.sum())*100.0, label = 'xgboost', color = colors_['ADn'])
# 	plot(bins2[0:-1], c2/float(c2.sum())*100.0, label = 'correlation', color = 'green')

# xlabel('Measure precision (a.u.)')
# ylabel('$\%$')


savefig("../../figures/fig4.pdf", dpi=900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig4.pdf &")
