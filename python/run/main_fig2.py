#!/usr/bin/env python

'''
	File name: main_fig2.py
	Author: Guillaume Viejo
	Date created: 27/03/2017    
	Python Version: 2.7

Fig 2

'''

import warnings
import pandas as pd
import scipy.io
import numpy as np
# Should not import fonctions if already using tensorflow for something else
from fonctions import *
import sys, os
import itertools
import cPickle as pickle


#####################################################################
# DATA LOADING
#####################################################################
# adrien_data = scipy.io.loadmat(os.path.expanduser('~/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData/data_test_boosted_tree_20ms.mat'))
# m1_imported = scipy.io.loadmat('/home/guillaume/spykesML/data/m1_stevenson_2011.mat')
adrien_data = scipy.io.loadmat(os.path.expanduser('../data/sessions/wake/boosted_tree.Mouse25-140130.mat'))

#####################################################################
# DATA ENGINEERING
#####################################################################
data 			= 	pd.DataFrame()
data['time'] 	= 	np.arange(len(adrien_data['Ang']))		# TODO : import real time from matlab script
data['ang'] 	= 	adrien_data['Ang'].flatten() 			# angular direction of the animal head
data['x'] 		= 	adrien_data['X'].flatten() 				# x position of the animal 
data['y'] 		= 	adrien_data['Y'].flatten() 				# y position of the animal
data['vel'] 	= 	adrien_data['speed'].flatten() 			# velocity of the animal 
# Engineering features
data['cos']		= 	np.cos(adrien_data['Ang'].flatten())	# cosinus of angular direction
data['sin']		= 	np.sin(adrien_data['Ang'].flatten())	# sinus of angular direction
# Firing data
for i in xrange(adrien_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = adrien_data['Pos'][:,i]
for i in xrange(adrien_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = adrien_data['ADn'][:,i]



#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
def extract_tree_order(trees, n_feat):
	# return array (len(trees), n_feat)	
	n = len(trees.get_dump())
	propor = np.zeros((n,n_feat))		
	for t in xrange(n):
		gv = xgb.to_graphviz(trees, num_trees=t)
		body = gv.body		
		for i in xrange(len(body)):
			for l in body[i].split('"'):
				if 'f' in l and '<' in l:
					feat = l.split("<")[0]
					propor[t,int(feat[1:])] += 1
	
	propor = np.cumsum(propor, 0)			
	propor = propor/np.vstack(np.sum(propor, 1))
	return propor

def extract_tree_threshold(trees):
	n = len(trees.get_dump())
	thr = {}
	for t in xrange(n):
		gv = xgb.to_graphviz(trees, num_trees=t)
		body = gv.body		
		for i in xrange(len(body)):
			for l in body[i].split('"'):
				if 'f' in l and '<' in l:
					tmp = l.split("<")
					if thr.has_key(tmp[0]):
						thr[tmp[0]].append(float(tmp[1]))
					else:
						thr[tmp[0]] = [float(tmp[1])]					
	for k in thr.iterkeys():
		thr[k] = np.sort(np.array(thr[k]))
	return thr

def tuning_curve(x, f, nb_bins, tau = 40.0):	
	bins = np.linspace(x.min(), x.max()+1e-8, nb_bins+1)
	index = np.digitize(x, bins).flatten()    
	tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)])  	
	occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)])
	tcurve = (tcurve/occupancy)*tau
	x = bins[0:-1] + (bins[1]-bins[0])/2.
	# tcurve = tcurve
	return (x, tcurve)

def fisher_information(x, f):
	fish = np.zeros(len(f)-1)
	slopes_ = []
	tmpf = np.hstack((f[-1],f,f[0:3]))
	binsize = x[1]-x[0]
	# very bad # to correct of the offset of x points
	tmpx = np.hstack((np.array([x[0]-binsize-(x.min()+(2*np.pi-x.max()))]),x,np.array([x[-1]+i*binsize+(x.min()+(2*np.pi-x.max())) for i in xrange(1,4)])))
	
	# plot(tmpx, tmpf, 'o')	
	for i in xrange(len(f)):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(tmpx[i:i+3], tmpf[i:i+3])
		slopes_.append(slope)	
		# plot(tmpx[i:i+3], tmpx[i:i+3]*slope+intercept, '-')

	fish = np.power(slopes_, 2)
	# for i in xrange(len(fish)):
	# 	fish[i] = np.power((f[i+1]-f[i])/binsize, 2)
	fish = fish/(f+1e-4)
	# fish = fish/f[0:-1]
	return (x, fish)

#####################################################################
# COMBINATIONS DEFINITION
#####################################################################
combination = {
	'ang':{
		'ADn':	{
			'features' 	:	['ang'],
			'targets'	:	['ADn.3']
				},
		'Pos':	{
			'features' 	:	['ang'],
			'targets'	:	['Pos.0']
				},		
		},
	'angxy': 	{
		'ADn':		{
			'features' 	:	['ang', 'x', 'y'],
			'targets'	:	['ADn.3']
					},
		'Pos':		{
			'features' 	:	['ang', 'x', 'y'],
			'targets'	:	['Pos.0']
					},				
				},
	'xy': 	{
		'ADn':		{
			'features' 	:	['x', 'y'],
			'targets'	:	['ADn.3']
					},
		'Pos':		{
			'features' 	:	['x', 'y'],
			'targets'	:	['Pos.0']
					},				
				}				
	}

#####################################################################
# LEARNING XGB Exemples
#####################################################################
params = {'objective': "count:poisson", #for poisson output
	'eval_metric': "logloss", #loglikelihood loss
	'seed': 2925, #for reproducibility
	'silent': 1,
	'learning_rate': 0.01,
	'min_child_weight': 2, 'n_estimators': 120,
	'max_depth': 5}        

num_round = 100
bsts = {}
for i in combination.iterkeys():
	bsts[i] = {}
	for j in combination[i].iterkeys():
		features = combination[i][j]['features']
		targets = combination[i][j]['targets']	
		X = data[features].values
		Yall = data[targets].values		
		for k in xrange(Yall.shape[1]):
			dtrain = xgb.DMatrix(X, label=Yall[:,k])
			bst = xgb.train(params, dtrain, num_round)
			bsts[i][j] = bst

# bsts = pickle.load(open("fig2_bsts.pickle", 'rb'))

#####################################################################
# TUNING CURVE
#####################################################################
X = data['ang'].values
example = [combination['ang'][k]['targets'][0] for k in ['ADn', 'Pos']]
tuningc = {}
for k in example:
	Y = data[k].values
	tuningc[k.split(".")[0]] = tuning_curve(X, Y, nb_bins = 60)


#####################################################################
# EXTRACT TREE STRUCTURE
#####################################################################
thresholds = {}
for i in bsts.iterkeys():
	thresholds[i] = {}
	for j in bsts[i].iterkeys():
		thresholds[i][j] = extract_tree_threshold(bsts[i][j])		

########################################################################
# DENSITY PICKLE LOAD
########################################################################
all_data = pickle.load(open("../data/fig2_density.pickle", 'rb'))
angdens = all_data['angdens']
mean_angdens = all_data['mean_angdens']
ratio = all_data['ratio']
twod = all_data['twod_xydens']
tcurves = all_data['alltcurve']
gain = all_data['gain']

########################################################################
# CORRELATION
########################################################################
corr = {'ADn':[], 'Pos':[]}
for g in corr.iterkeys():
	for k in angdens['1.'+g].iterkeys():
		tun = tcurves[k] # need to center tun 
		offset = tun[0][np.argmax(tun[1])]
		tun[0] -= offset
		tun[0][tun[0] <= -np.pi] += 2*np.pi
		tun[0][tun[0] > np.pi] -= 2*np.pi
		tun[1] = tun[1][np.argsort(tun[0])]
		tun[0] = np.sort(tun[0])		
		fis = fisher_information(tun[0], tun[1])[1] 
		# fis = np.hstack((fis, fis[0])) # looping		
		fis = fis.reshape(20,3).mean(1)
		dens = angdens['1.'+g][k][1]
		corr[g].append(scipy.stats.pearsonr(fis, dens)[0])
		# if corr[g][-1] < 0.0:
		# 	sys.exit()


########################################################################
# PLOTTING
########################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.1            # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)


import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	"text.usetex": True,                # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 7,               # LaTeX default is 10pt font.
	"font.size": 8,
	"legend.fontsize": 5,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 5,
	"ytick.labelsize": 5,
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
from mpl_toolkits.axes_grid.inset_locator import inset_axes


trans = {'f0':'Angle','f1':'x pos','f2':'y pos'}
colors_ = ['#EE6C4D', '#3D5A80']
title_ = ['Antero-dorsal nucleus', 'Post-subiculum']







figure(figsize = figsize(1))
# subplots_adjust(hspace = 0.4, right = 0.4)
# outer = gridspec.GridSpec(2,2, height_ratios = [1, 0.6])
outer = gridspec.GridSpec(2,2, width_ratios = [1.6,0.7], wspace = 0.2, hspace = 0.3)

##PLOT 1#################################################################################################################
# Examples subplot 1 et 2
#########################################################################################################################
gs1 	= gridspec.GridSpec(1,2)
gs1.update(wspace = 0.4, bottom = 0.65, top = 1, right = 0.6, left = 0.04)

limts = [(1.5,5),(0.7, 3.6)]
for e, i in zip(['ADn','Pos'],range(2)):			
	ax = subplot(gs1[0,i])
	simpleaxis(ax)	
	[ax.axvline(l, alpha = 0.9, color = 'grey', linewidth = 0.2) for l in np.unique(thresholds['ang'][e]['f0'])[0:100]]	
	fisher = fisher_information(tuningc[e][0], tuningc[e][1])
	ax2 = ax.twinx()
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)	
	ax2.plot(fisher[0], fisher[1], '-', color = '#25393C', label = 'Fisher \n Information', linewidth = 1.0)
	ax2.set_yticks([])
	ax.plot(tuningc[e][0], tuningc[e][1], color = colors_[i], linewidth = 1.5)
	ax.set_xlim(limts[i])
	# ax.set_xticks([0, 2*np.pi])
	# ax.set_xticklabels(('0', '$2\pi$'))
	ax.set_xlabel('Head-direction (rad)', labelpad = 0.0)
	ax.set_ylabel('Firing rate (Hz)', labelpad = 2.1)		
	ax.locator_params(axis='y', nbins = 3)
	ax.locator_params(axis='x', nbins = 4)
	if j == 0:
		ax.set_title(title_[i], loc = 'right')
	leg = ax2.legend(fontsize = 5, loc = 'best')
	leg.get_frame().set_linewidth(0.0)
	# leg.get_frame().set_facecolor('white')
	ax.set_title(title_[i], fontsize = 7)

##PLOT 2#################################################################################################################
# Centered density
#########################################################################################################################
gs2 	= gridspec.GridSpec(1,2)
gs2.update(wspace = 0.4, bottom = 0.0, top = 0.55, right = 0.6, left = 0.04)
tmp = []
for g,i in zip(['1.ADn', '1.Pos'], xrange(2)):
	ax = subplot(gs2[0,i])	
	simpleaxis(ax)
	for k in angdens[g].iterkeys():
		if np.max(angdens[g][k][1]*100.0) < 50:
			plot(angdens[g][k][0], angdens[g][k][1]*100.0, '-', color = colors_[i], linewidth = 0.4, alpha = 0.1)

	plot(mean_angdens[g][0], mean_angdens[g][1]*100.0, '-', color = colors_[i], linewidth = 1.2, alpha = 1)		
	axhline(5, linestyle = '--', color = 'black', linewidth=0.6)
	ylabel("Density of angular splits $(\%)$")
	xlim(-np.pi, np.pi)
	xticks([-np.pi, 0, np.pi], ('$-\pi$', '0', '$\pi$'))
	xlabel('Centered', labelpad = 0.4)
	locator_params(axis='y', nbins = 5)
	locator_params(axis='x', nbins = 5)
	#inset	
	tmp.append(ax.get_position().bounds)


ai = axes([tmp[0][0]+tmp[0][2]*0.7,tmp[0][1]+tmp[0][3]*0.8, 0.09, 0.12])
ai.get_xaxis().tick_bottom()
ai.get_yaxis().tick_left()
n, bins, patches = ai.hist(corr['ADn'], 8, normed = 1, facecolor = 'white', edgecolor = colors_[0])
ai.set_xlim(-1, 1)
ai.set_xticks([-1, 0, 1])
ai.set_xticklabels([-1,'',1])
ai.set_yticks([])
ai.set_title(r'$<$Fisher,Splits$>$', fontsize = 6, position = (0.5, 0.9))
ai.set_xlabel("$r$", fontsize = 6, labelpad = -2.4)
ai.set_ylabel("$\%$", fontsize = 6, labelpad = 0.5)

aii = axes([tmp[1][0]+tmp[1][2]*0.7,tmp[1][1]+tmp[1][3]*0.8, 0.09, 0.12])
aii.get_xaxis().tick_bottom()
aii.get_yaxis().tick_left()
n, bins, patches = aii.hist(corr['Pos'], 8, normed = 1, facecolor = 'white', edgecolor = colors_[1])
aii.set_xlim(-1,1)
aii.set_xticks([-1, 0, 1])
aii.set_xticklabels([-1,'',1])
aii.set_yticks([])
aii.set_title(r'$<$Fisher,Splits$>$', fontsize = 6, position = (0.5, 0.9))
aii.set_xlabel("$r$", fontsize = 6, labelpad = -2.4)
aii.set_ylabel("$\%$", fontsize = 6, labelpad = 0.5)







##PLOT 3#################################################################################################################
# x y split
#########################################################################################################################
gs3 	= gridspec.GridSpec(2,2)
gs3.update(wspace = 0.4, hspace = 0.4, bottom = 0.65, top = 1, right = 1.0, left = 0.7)

title2 = ['ADn', 'PoSub']
for e, i in zip(['ADn','Pos'],range(2)):			
	ax = subplot(gs3[0,i])
	simpleaxis(ax)	

	[ax.axvline(l, alpha = 0.6, color = colors_[i], linewidth = 0.2) for l in np.unique(thresholds['xy'][e]['f0'])]
	[ax.axhline(l, alpha = 0.6, color = colors_[i], linewidth = 0.2) for l in np.unique(thresholds['xy'][e]['f1'])]	
	ax.plot(data['x'].values[0:6000], data['y'].values[0:6000], '-', color = 'grey', alpha = 1, linewidth = 0.4)				
	ax.set_xlabel('x pos', labelpad = 0.4)
	ax.set_ylabel('y pos')
	ax.set_xticks([])
	ax.set_yticks([])	
	ax.set_xlim(thresholds['xy'][e]['f0'].min(), thresholds['xy'][e]['f0'].max())
	ax.set_ylim(thresholds['xy'][e]['f1'].min(), thresholds['xy'][e]['f1'].max())
	ax.set_title(title2[i], fontsize = 7)

count = 0
for e, i in zip(['ADn','Pos'],range(2,4)):			
	ax = subplot(gs3[1,count])
	simpleaxis(ax)	
	im = ax.imshow(twod['3.'+e].transpose(), origin = 'lower', interpolation = 'nearest', aspect=  'auto', cmap = "gist_yarg")
	ax.set_xlabel('x pos', labelpad = 0.4)
	ax.set_ylabel('y pos')
	ax.set_xticks([])
	ax.set_yticks([])		
	count += 1
#  	cbar = colorbar(im, orientation = 'horizontal', fraction = 0.05, pad = 0.4, ticks=[np.min(twod['3.'+e]), np.max(twod['3.'+e])])
# 	cbar.set_ticklabels([np.min(twod['3.'+e]), np.max(twod['3.'+e])])
# 	cbar.ax.tick_params(labelsize = 4)
# 	cbar.update_ticks()

ax.set_title('Density of (x,y) splits $(\%)$', fontsize=7, position = (-0.35, 0.95))

##PLOT 4#################################################################################################################
# ang x y density
#########################################################################################################################
gs4 	= gridspec.GridSpec(2,2)
gs4.update(wspace = 0.3, hspace = 0.06, bottom = 0.35, top = 0.55, right = 1.0, left = 0.7)

labels = ['ADn', 'PoSub']

subplot(gs4[:,0])
simpleaxis(gca())
x = np.arange(3, dtype = float)
for g, i in zip(ratio.iterkeys(), xrange(2)):
	tmp = []
	for k in ratio[g].iterkeys():		
		tmp.append(ratio[g][k])
	tmp = np.array(tmp)*100
	mean = tmp.mean(0)
	sem = np.std(tmp, 0)/np.sqrt(np.size(tmp))
	bar(x, mean, 0.4, yerr = sem, align='center',
		ecolor='k', alpha=.9, color=colors_[i], ec='w', label = labels[i], capsize = 1)
	plot(x, mean, 'o', alpha = 1, color = 'black', markersize = 2)
	x += 0.41
	# xticks([])
	# yticks([])
# leg = legend(loc = 'upper right')
# leg.get_frame().set_linewidth(0.0)
locator_params(axis='y', nbins = 3)
ylabel('$(\%)$')
xticks(np.arange(3)+0.205, ('Angle','x pos', 'y pos'), rotation = 45, ha='right')
title('Density\nof splits', fontsize = 7)


subplot(gs4[0,1])
ax = gca()
simpleaxis(ax)
x = np.arange(3, dtype = float)
for g, i in zip(gain.iterkeys(), xrange(2)):
	tmp = []
	for k in gain[g].iterkeys():		
		tmp.append(gain[g][k])
	tmp = np.array(tmp)*100
	mean = tmp.mean(0)	
	sem = np.std(tmp, 0)/np.sqrt(np.size(tmp))
	bar(x, mean, 0.4, yerr = sem, align='center',
		ecolor='k', alpha=.9, color=colors_[i], ec='w', label = labels[i], capsize = 1)
	plot(x, mean, 'o', alpha = 1, color = 'black', markersize = 2)
	x += 0.41
	xticks([])
	yticks([70,90])
	# leg = legend()
	# leg.get_frame().set_linewidth(0.0)
	ylim(60, 90)
	xlim(-0.5, 2.95)
leg = legend(fontsize = 5, bbox_to_anchor = (0,1))
leg.get_frame().set_linewidth(0.0)		
locator_params(axis='y', nbins = 2)
# ylabel('$(\%)$')
ax.spines['bottom'].set_visible(False)
title('Gain', fontsize = 7)
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal


subplot(gs4[1,1])
ax = gca()
simpleaxis(ax)
x = np.arange(3, dtype = float)
for g, i in zip(gain.iterkeys(), xrange(2)):
	tmp = []
	for k in gain[g].iterkeys():		
		tmp.append(gain[g][k])
	tmp = np.array(tmp)*100
	mean = tmp.mean(0)	
	sem = np.std(tmp, 0)/np.sqrt(np.size(tmp))
	bar(x, mean, 0.4, yerr = sem, align='center',
		ecolor='k', alpha=.9, color=colors_[i], ec='w', label = labels[i], capsize = 1)
	plot(x, mean, 'o', alpha = 1, color = 'black', markersize = 2)
	x += 0.41
	# xticks([])
	# yticks([])
	# leg = legend()
	# leg.get_frame().set_linewidth(0.0)
	ylim(0,18)
	xlim(-0.5, 2.95)
xticks(np.arange(3)+0.205, ('Angle','x pos', 'y pos'), rotation = 45, ha='right')
locator_params(axis='y', nbins = 3)
kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal


##PLOT 5#################################################################################################################
# POISSON PROCESS
#########################################################################################################################
store 	= pd.HDFStore("../data/fig2_brian_gain.h5")
brian_gain = store['data']
store.close()

brian_pos = brian_gain.filter(like = 'Pos', axis =  0)
brian_adn = brian_gain.filter(like = 'ADn', axis =  0)


gs5 	= gridspec.GridSpec(2,2)
gs5.update(wspace = 0.3, hspace = 0.06, bottom = 0.0, top = 0.23, right = 1.0, left = 0.7)

labels = ['ADn', 'PoSub']



subplot(gs5[:,0])
simpleaxis(gca())
x = np.arange(3, dtype = float)
order = ['ang.ang', 'ang.x', 'ang.y']
for g, i in zip(['ADn', 'Pos'], xrange(2)):
	tmp = brian_gain.filter(like = g, axis =  0)
	mean = tmp.mean(0).loc[order]
	sem = tmp.sem(0).loc[order]
	bar(x, mean, 0.4, yerr = sem, align='center',
		ecolor='k', alpha=.9, color=colors_[i], ec='w', label = labels[i], capsize = 1)
	plot(x, mean, 'o', alpha = 1, color = 'black', markersize = 2)
	x += 0.41

locator_params(axis='y', nbins = 3)
ylabel('Gain')
xticks(np.arange(3)+0.205, ('Angle','x pos', 'y pos'), rotation = 45, ha='right')
title('Sampling from \n angular tuning curve', fontsize = 7)
ylim(0, 25)

# subplot(gs4[0,1])
# ax = gca()
# simpleaxis(ax)
# x = np.arange(3, dtype = float)
# for g, i in zip(gain.iterkeys(), xrange(2)):
# 	tmp = []
# 	for k in gain[g].iterkeys():		
# 		tmp.append(gain[g][k])
# 	tmp = np.array(tmp)*100
# 	mean = tmp.mean(0)	
# 	sem = np.std(tmp, 0)/np.sqrt(np.size(tmp))
# 	bar(x, mean, 0.4, yerr = sem, align='center',
# 		ecolor='k', alpha=.9, color=colors_[i], ec='w', label = labels[i], capsize = 1)
# 	plot(x, mean, 'o', alpha = 1, color = 'black', markersize = 2)
# 	x += 0.41
# 	xticks([])
# 	yticks([70,90])
# 	# leg = legend()
# 	# leg.get_frame().set_linewidth(0.0)
# 	ylim(60, 90)
# 	xlim(-0.5, 2.95)
# leg = legend(fontsize = 5, bbox_to_anchor = (0,1))
# leg.get_frame().set_linewidth(0.0)		
# locator_params(axis='y', nbins = 2)
# # ylabel('$(\%)$')
# ax.spines['bottom'].set_visible(False)
# title('Gain', fontsize = 7)
# d = .015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal


subplot(gs5[:,1])
ax = gca()
simpleaxis(ax)
x = np.arange(3, dtype = float)
order = ['spc.ang', 'spc.x', 'spc.y']
for g, i in zip(['ADn', 'Pos'], xrange(2)):
	tmp = brian_gain.filter(like = g, axis =  0)
	mean = tmp.mean(0).loc[order]
	sem = tmp.sem(0).loc[order]
	bar(x, mean, 0.4, yerr = sem, align='center',
		ecolor='k', alpha=.9, color=colors_[i], ec='w', label = labels[i], capsize = 1)
	plot(x, mean, 'o', alpha = 1, color = 'black', markersize = 2)
	x += 0.41
	# xticks([])
	# yticks([])
	# leg = legend()
	# leg.get_frame().set_linewidth(0.0)
	# ylim(0,18)
	# xlim(-0.5, 2.95)
xticks(np.arange(3)+0.205, ('Angle','x pos', 'y pos'), rotation = 45, ha='right')
locator_params(axis='y', nbins = 3)
# kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
# ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
title('spatial tuning curve', fontsize = 7)
ylabel('Gain')
ylim(0, 25)



savefig("../../figures/fig2.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig2.pdf &")


