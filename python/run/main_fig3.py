#!/usr/bin/env python

'''
	File name: main_fig4.py
	Author: Guillaume Viejo
	Date created: 30/03/2017    
	Python Version: 2.7


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
# DATA LOADING | ALL SESSIONS WAKE
#####################################################################
# thresholds dict from z620 in ../data/results_peer_fig3/
# good pr2 with 100 trees in ../data/results_peer_fig3_good_100trees




thr_directory = "../data/results_peer_fig3"
pr2_directory = "../data/results_peer_fig3"
gain_directory = "../data/results_peer_fig3"
loo_directory = "../data/results_peer_fig3"
equal_directory = "../data/results_peer_fig3"
# THRESHOLDS
data_thr = {}
for ep in ['wake', 'rem', 'sws']:
	# os.system("scp viejo@guillimin.hpc.mcgill.ca:~/results_peer_fig3/"+ep+"/peer_bsts* ../data/results_peer_fig3/"+ep+"/")
	data_thr[ep] = {}
	for f in os.listdir(thr_directory+"/"+ep+"/"):
		if 'bsts' in f:
			data_thr[ep][f] = pickle.load(open(thr_directory+"/"+ep+"/"+f, 'rb'))

# PR2
data = {}
# TO RECHANGE WHEN IT"S DONE IN Z620
for ep in ['wake', 'rem', 'sws']:
	data[ep] = {}
	for f in os.listdir(pr2_directory+"/"+ep+"/"):
		if 'pr2' in f:
			data[ep][f.split(".")[1]] = pickle.load(open(pr2_directory+"/"+ep+"/"+f, 'rb'))

data_pr2 = {}
for g in ['ADn', 'Pos']:
	data_pr2[g] = {}
	for w in ['peer', 'cros']:
		data_pr2[g][w] = {}
		for e in ['wake', 'rem', 'sws']:
			data_pr2[g][w][e] = []
			for s in data[e].iterkeys():
				data_pr2[g][w][e].append(data[e][s][g][w]['PR2'])
			data_pr2[g][w][e] = np.vstack(data_pr2[g][w][e])
# pr2_sleep = pickle.load(open("../data/fig3_pr2_sleep.pickle", 'rb'))

# CORRELATION
# os.system("scp -r viejo@guillimin.hpc.mcgill.ca:~/results_peer_fig3/wake/peer_corr* ../data/results_peer_fig3/wake/")
corr = {}
for g in ['ADn', 'Pos']:
	corr[g] = {}
	for f in os.listdir(pr2_directory+"/wake/"):
		if 'corr' in f:
			tmp = pickle.load(open(pr2_directory+"/wake/"+f, 'rb'))
			for k in tmp[g]['peer']['corr'].keys():
				corr[g][k] = tmp[g]['peer']['corr'][k]

# GAIN
# os.system("scp viejo@guillimin.hpc.mcgill.ca:~/results_peer_fig3/wake/peer_gain* ../data/results_peer_fig3/wake/")
datagain = {}
for f in os.listdir(gain_directory+"/wake/"):
	if 'gain' in f:
		datagain[f] = pickle.load(open(gain_directory+"/wake/"+f, 'rb'))			

# LOO
# os.system("scp viejo@guillimin.hpc.mcgill.ca:~/results_peer_fig3/wake/peer_loo* ../data/results_peer_fig3/wake/")
dataloo = {}
for f in os.listdir(loo_directory+"/wake/"):
	if 'loo' in f:
		dataloo[f] = pickle.load(open(gain_directory+"/wake/"+f, 'rb'))			

# EQUAL
# os.system("scp viejo@guillimin.hpc.mcgill.ca:~/results_peer_fig3/wake/peer_equal* ../data/results_peer_fig3/wake/")
dataequal = {}
for f in os.listdir(equal_directory+"/wake/"):
	if 'equal' in f:
		dataequal[f] = pickle.load(open(equal_directory+"/wake/"+f, 'rb'))

# #####################################################################
# # TUNING CURVE
# #####################################################################
tuningc = {}
for f in os.listdir("../data/results_density/wake/"):
	tmp = pickle.load(open("../data/results_density/wake/"+f))
	tmp = tmp['tuni']
	for k in tmp.iterkeys():
		tuningc[k] = tmp[k]	

#####################################################################
# EXTRACT TREE STRUCTURE
#####################################################################
names = pickle.load(open("../data/results_peer_fig3/fig3_names.pickle", 'rb'))

thresholds = {}
gain = {}
for g in ['ADn', 'Pos']:
	thresholds[g] = {}
	gain[g] = {}
	for w in ['peer', 'cros']:
		thresholds[g][w] = {}
		gain[g][w] = {}
		for s in data_thr['wake'].iterkeys(): # sessions		
			for k in data_thr['wake'][s][g][w].iterkeys():		
				thresholds[g][w][k] = data_thr['wake'][s][g][w][k]
		for s in datagain.iterkeys():
			for k in datagain[s][g][w].iterkeys():		
				gain[g][w][k] = datagain[s][g][w][k]



# need to sort the features by the number of splits
sorted_features = {}
sorted_gain = {}
for g in thresholds.iterkeys():
	sorted_features[g] = {}
	sorted_gain[g] = {}
	for w in thresholds[g].iterkeys():
		sorted_features[g][w] = {}
		sorted_gain[g][w] = {}
		for k in thresholds[g][w].iterkeys(): # PREDICTED NEURONS			
			count = np.array([len(thresholds[g][w][k][f]) for f in thresholds[g][w][k].iterkeys()])
			name = np.array([names[g][w][k][int(f[1:])] for f in thresholds[g][w][k].iterkeys()])
			sorted_features[g][w][k] = np.array([name[np.argsort(count)], np.sort(count)])
			gain_ = np.array([gain[g][w][k][f] for f in gain[g][w][k].iterkeys()])
			name = np.array([names[g][w][k][int(f[1:])] for f in gain[g][w][k].iterkeys()])
			sorted_gain[g][w][k] = np.array([name[np.argsort(gain_)], np.sort(gain_)])



#####################################################################
# number of splits versus mean firing rate
#####################################################################
splitvar = {}
plotsplitvar = {}
for g in thresholds.iterkeys():
	splitvar[g] = {}
	plotsplitvar[g] = {}
	for w in thresholds[g].iterkeys():
		splitvar[g][w] = {}
		plotsplitvar[g][w] = {'nsplit':[], 'meanf':[], 'meanfdiff':[]}
		for k in thresholds[g][w].iterkeys():
			mean_firing_rate = []
			mean_firing_rate_diff = []
			for n in sorted_features[g][w][k][0]:
				mean_firing_rate.append(np.mean(tuningc[n][1]))
				mean_firing_rate_diff.append(np.mean(tuningc[k][1])-np.mean(tuningc[n][1]))
			mean_firing_rate = np.array(mean_firing_rate)
			mean_firing_rate_diff = np.array(mean_firing_rate_diff)			
			splitvar[g][w][k] = np.array([mean_firing_rate, sorted_features[g][w][k][1]])
			plotsplitvar[g][w]['meanf'].append(mean_firing_rate)
			plotsplitvar[g][w]['meanfdiff'].append(mean_firing_rate_diff)
			plotsplitvar[g][w]['nsplit'].append(sorted_features[g][w][k][1].astype('float'))
		plotsplitvar[g][w]['meanf'] = np.hstack(np.array(plotsplitvar[g][w]['meanf']))
		plotsplitvar[g][w]['nsplit'] = np.hstack(np.array(plotsplitvar[g][w]['nsplit']))
		plotsplitvar[g][w]['meanfdiff'] = np.hstack(np.array(plotsplitvar[g][w]['meanfdiff']))
		

#####################################################################
# DISTANCE TO CENTER OF FIELD
#####################################################################
distance = {}
plotdistance = {}
for g in thresholds.iterkeys():
	distance[g] = {}
	plotdistance[g] = {}
	for w in thresholds[g].iterkeys():
		distance[g][w] = {}
		plotdistance[g][w] = {'nsplit':[], 'distance':[]}
		for k in thresholds[g][w].iterkeys():
			com_neuron = tuningc[k][0][np.argmax(tuningc[k][1])]				
			com = np.array([tuningc[n][0][np.argmax(tuningc[n][1])] for n in sorted_features[g][w][k][0]])			
			dist = np.abs(com - com_neuron)
			tmp = 2*np.pi - dist[dist>np.pi]
			dist[dist>np.pi] = tmp
			plotdistance[g][w]['distance'].append(dist)
			plotdistance[g][w]['nsplit'].append(sorted_features[g][w][k][1].astype('float'))
		plotdistance[g][w]['distance'] = np.hstack(np.array(plotdistance[g][w]['distance']))
		plotdistance[g][w]['nsplit'] = np.hstack(np.array(plotdistance[g][w]['nsplit']))

#####################################################################
# PEER CORRELATION 
#####################################################################
peercorr = {}
for g in corr.iterkeys():
	peercorr[g] = []
	for k in corr[g].iterkeys():
		for n,i in zip(corr[g][k][0], xrange(len(corr[g][k][0]))):
			comn = tuningc[n][0][np.argmax(tuningc[n][1])]
			comk = tuningc[k][0][np.argmax(tuningc[k][1])]
			dist = np.abs(comn - comk) 
			if dist > np.pi: dist = 2*np.pi - dist
			if comn < comk: dist *= -1.0
			peercorr[g].append(np.array([dist, float(corr[g][k][1][i])]))
	peercorr[g] = np.array(peercorr[g])

#####################################################################
# LEAVE ONE OUT
#####################################################################
loo = {}
for g in ['ADn', 'Pos']:
	loo[g] = {}
	for s in dataloo.iterkeys():
		for k in dataloo[s][g]['peer'].iterkeys():
			for n in dataloo[s][g]['peer'][k].iterkeys():
				if n in loo[g].keys():
					loo[g][n].append(dataloo[s][g]['peer'][k][n].flatten())
				else:
					loo[g][n] = [dataloo[s][g]['peer'][k][n].flatten()]

meanloo = {}
for g in loo.keys():
	meanloo[g] = []
	for n in loo[g].iterkeys():
		loo[g][n] = np.hstack(loo[g][n])
		meanloo[g].append([np.mean(loo[g][n]), scipy.stats.sem(loo[g][n])])
	meanloo[g] = np.array(meanloo[g])

#####################################################################
# EQUAL
#####################################################################
equal = {}
for g in ['ADn', 'Pos']:
	equal[g] = {}	
	for w in ['peer', 'cros']:
		equal[g][w] = {}
		for s in dataequal.iterkeys():		
			se = s.split(".")[1]
			equal[g][w][se] = []
			if g in dataequal[s].keys():			
				if w in dataequal[s][g].keys():									
					for k in dataequal[s][g][w].iterkeys():
						equal[g][w][se].append(dataequal[s][g][w][k])
			equal[g][w][se] = np.array(equal[g][w][se])

meanequal = {}
for w in ['peer', 'cros']:
	meanequal[w] = {}
	for s in dataequal.keys():
		se = s.split(".")[1]
		meanequal[w][se] = []
		for g in ['ADn', 'Pos']:
			meanequal[w][se].append(np.mean(equal[g][w][se]))

#####################################################################
# TIME SPLIT LOADING
#####################################################################
time_data = pickle.load(open("../data/fig3_timesplit.pickle", 'rb'))

#####################################################################
# PLOTTING
#####################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean *0.9             # height in inches
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
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
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



















figure(figsize = figsize(1))
outerspace = gridspec.GridSpec(1,2, width_ratios =[1.2,0.9], wspace = 0.3)

#################################################################
# LEFT
#################################################################
# outer = gridspec.GridSpec(outerspace[0], height_ratios=[0.5,1.2])

# SUBPLOT 1 ################################################################
# outer = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec = outerspace[0], height_ratios=[1, 0.8], hspace = 0.3)

gs = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = outerspace[0], height_ratios=[1.1, 0.5], hspace = 0.4, wspace = 0.5)

subplot(gs[0,:])
simpleaxis(gca())


y = []
err = []
x = [0.0]
color = []
for w in ['peer', 'cros']:		
	for g in ['ADn', 'Pos']:            	
		for e in ['wake', 'rem', 'sws']:
			PR2_art = data_pr2[g][w][e]
			color.append(colors_[g])
			y.append(np.mean(PR2_art))
			err.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))
			x.append(x[-1]+0.42)
		
		
		x[-1] += 0.3	
	x[-1] += 0.4
		
x = np.array(x)[0:-1]
y = np.array(y)
err = np.array(err)		

ind_adn = [0,1,2,6,7,8]
ind_pos = [3,4,5,9,10,11]
x_adn = x[ind_adn]
y_adn = y[ind_adn]
e_adn = err[ind_adn]
x_pos = x[ind_pos]
y_pos = y[ind_pos]
e_pos = err[ind_pos]

ind = [0,3]
bar(x_adn[ind], y_adn[ind], 0.4, align='center',
			ecolor='k', color = colors_['ADn'], alpha=1, ec='w', yerr=e_adn[ind], label = 'Antero Dorsal')
bar(x_pos[ind], y_pos[ind], 0.4, align='center',
			ecolor='k', color = colors_['Pos'], alpha=1, ec='w', yerr=e_pos[ind], label = 'Post Subiculum')
ind = [1,4]
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = 'white', edgecolor='black', alpha=1, hatch="//////", linewidth = 0, label = 'REM sleep')
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = colors_['ADn'], edgecolor='black', alpha=1, hatch="//////", linewidth = 0)
bar(x_pos[ind], y_pos[ind], 0.4, align='center', facecolor = colors_['Pos'], edgecolor='black', alpha=1, hatch="//////", linewidth = 0)
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = 'none', alpha=1, edgecolor='w', yerr=e_adn[ind], ecolor = 'black')
bar(x_pos[ind], y_pos[ind], 0.4, align='center', facecolor = 'none', alpha=1, edgecolor='w', yerr=e_pos[ind], ecolor = 'black')

ind = [2,5]
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = 'white', edgecolor='black', alpha=1, hatch="xxxx", linewidth = 0, label = 'Slow wave sleep')
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = colors_['ADn'], edgecolor='black', alpha=1, hatch="xxxx", linewidth = 0)
bar(x_pos[ind], y_pos[ind], 0.4, align='center', facecolor = colors_['Pos'], edgecolor='black', alpha=1, hatch="xxxx", linewidth = 0)
bar(x_adn[ind], y_adn[ind], 0.4, align='center', facecolor = 'none', alpha=1, edgecolor='w', yerr=e_adn[ind], ecolor = 'black')
bar(x_pos[ind], y_pos[ind], 0.4, align='center', facecolor = 'none', alpha=1, edgecolor='w', yerr=e_pos[ind], ecolor = 'black')


plot(x, y, 'k.', markersize=3)         
locator_params(nbins=4)				
xlim(np.min(x)-0.3, np.max(x)+0.3)
ylabel('Pseudo-R2 (XGBoost)')
xticks(x[[1,4,7,10]], 
	["ADn$\Rightarrow$ADn", "PoSub$\Rightarrow$PoSub", "PoSub$\Rightarrow$ADn", "ADn$\Rightarrow$PoSub"], 
	# rotation = 30, 
	# ha = 'right'
	fontsize = 5
	)

legend(bbox_to_anchor=(0.55, 1.15), loc='upper center', ncol=2, frameon = False, columnspacing = 0.6)

title2 = ['WITHIN', 'BETWEEN']
count = 0
labels2 = {'peer':['ADn$\Rightarrow$ADn', 'PoSub$\Rightarrow$PoSub'],'cros':['ADn$\Rightarrow$PoSub', 'PoSub$\Rightarrow$ADn']}
for w in ['peer', 'cros']:
	subplot(gs[1,count])
	simpleaxis(gca())	
	for s in meanequal[w].iterkeys():	
		plot([0], meanequal[w][s][0], 'o', color=colors_['ADn'], markersize = 4)
		plot([1], meanequal[w][s][1], 'o', color=colors_['Pos'], markersize = 4)
		plot([0,1], meanequal[w][s], '-', color = 'grey')

	xticks(fontsize = 4)
	yticks(fontsize = 4)		
	# xlabel("Number of neurons", fontsize = 5, labelpad = 0.5)
	ylabel("p-$R^2$", fontsize = 6)
	xticks([0, 1], labels2[w], fontsize = 5)
	xlim(-0.4, 1.4)
	title(title2[count], fontsize = 6, y = 1.1)
	count += 1
	ylim(0, 0.8)
	locator_params(axis='y', nbins = 5)

# figtext(0.2, -0.2, "ADn $\Rightarrow$ ADn \n Post-S $\Rightarrow$ Post-S \n \scriptsize{(Features $\Rightarrow$ Target)}")
# figtext(0.6, -0.14, "ADn $\Rightarrow$ Post-S \n Post-S $\Rightarrow$ ADn")


#################################################################
# RIGHT
#################################################################
matplotlib.rcParams.update({"axes.labelsize": 	7,
							"font.size": 		8,
							"legend.fontsize": 	8,
							"xtick.labelsize": 	5,
							"ytick.labelsize": 	5,   
							})               # Make the legend/label fonts a little smaller
outer = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = outerspace[1], hspace = 0.3, wspace = 0.5)

count = 0

title_ = ["ADn $\Rightarrow$ ADn \n(wake)", "PoSub $\Rightarrow$ PoSub \n(wake)"]							


	

for g in plotsplitvar.keys():
	for w in ['peer']:
		subplot(outer[count])
		simpleaxis(gca())
		plot(plotdistance[g][w]['distance'], plotdistance[g][w]['nsplit'], 'o', color = colors_[g], markersize = 1)
		# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(plotdistance[g][w]['distance'], plotdistance[g][w]['nsplit'])
		# print r_value, p_value
		# x = np.array([np.min(plotdistance[g][w]['distance']), np.max(plotdistance[g][w]['distance'])])
		# plot(x, x*slope + intercept, '-', color = 'black', linewidth = 0.7)
		x, y = (plotdistance[g][w]['distance'], plotsplitvar[g][w]['nsplit'])
		nb_bins=5
		bins = np.linspace(x.min(), x.max()+1e-8, nb_bins+1)
		index = np.digitize(x, bins).flatten()
		curve = np.array([np.mean(y[index == i]) for i in xrange(1, nb_bins+1)])
		xx = bins[0:-1] + (bins[1]-bins[0])/2.
		plot(xx, curve, 'o-', color = 'black', linewidth = 0.8, markersize = 2.0) 
		# ax2.set_yticks([], [])			
		locator_params(nbins=2)				
		# ticklabel_format(style='sci', axis='x', scilimits=(0,0), fontsize = 4)
		# ticklabel_format(style='sci', axis='y', scilimits=(0,0), fontsize = 4)
		xticks([0, np.pi], ['0', '$\pi$'], fontsize = 4)
		yticks(fontsize = 4)		
		xlabel("Angular distance", fontsize = 6, labelpad = 0.4)				
		ylabel("Number of splits", fontsize = 7, labelpad = 0.6)
		xlim(0, np.pi)
		ylim(0,)
		title(title_[count-2], fontsize = 7)#, loc = 'left', y = 1.3)		


		subplot(outer[count+2])
		simpleaxis(gca())		
		plot(plotsplitvar[g][w]['meanf'], plotsplitvar[g][w]['nsplit'], 'o', color = colors_[g], markersize = 1)
		# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(plotsplitvar[g][w]['meanf'], plotsplitvar[g][w]['nsplit'])
		# x = np.array([np.min(plotsplitvar[g][w]['meanf']), np.max(plotsplitvar[g][w]['meanf'])])
		# plot(x, x*slope + intercept, '-', color = 'black', linewidth = 0.7)
		# print r_value, p_value
		xticks(fontsize = 4)
		yticks(fontsize = 4)		
		xlabel("Firing rate (Hz)", fontsize = 6, labelpad = 0.8)
		ylabel("Number of splits", fontsize = 7, labelpad = 0.6)		
		x, y = (plotsplitvar[g][w]['meanf'], plotsplitvar[g][w]['nsplit'])
		nb_bins=5
		bins = np.linspace(x.min(), x.max()+1e-8, nb_bins+1)
		index = np.digitize(x, bins).flatten()
		curve = np.array([np.mean(y[index == i]) for i in xrange(1, nb_bins+1)])
		
		xx = bins[0:-1] + (bins[1]-bins[0])/2.

		plot(xx, curve, 'o-', color = 'black', linewidth = 0.7, markersize = 2.0) 
		
		locator_params(axis='y', nbins = 5)
		
		



		count += 1




savefig("../../figures/fig3.pdf", dpi=900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig3.pdf &")
