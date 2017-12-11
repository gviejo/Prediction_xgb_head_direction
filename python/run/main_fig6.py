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

#####################################################################
# BRIAN NETWORKS
#####################################################################
def tuning_curve(x, f, nb_bins, tau = 40.0):	
	bins = np.linspace(0, 2*np.pi, nb_bins+1)
	index = np.digitize(x, bins).flatten()    
	tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)]).astype('float')  	
	occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)]).astype('float')
	
	tcurve = (tcurve/occupancy)*tau	
	x = bins[0:-1] + (bins[1]-bins[0])/2.
	
	return (x, tcurve, index-1)

def get_tuning_curves(n, x):
	# phi = np.sort(np.random.uniform(0, 2*np.pi, n))
	phi = np.arange(0, 2*np.pi, np.pi/2.)
	phi += np.random.rand(4)*1
	A 	= np.random.uniform(10, 50,n)
	B 	= np.random.uniform(5, 10, n)
	C 	= np.random.uniform(0, 3, n)
	return pd.DataFrame(index = x, columns = np.arange(n), data = C+A*np.exp(B*np.cos(np.vstack(x) - phi))/np.exp(B))

bin_size = 25
ang = scipy.io.loadmat("../data/sessions_nosmoothing_"+str(bin_size)+"ms/wake/boosted_tree.Mouse28-140313.mat")['Ang'].flatten()
ang = ang[30000:30600]
n_repeat 		= 1
nang 			= np.tile(ang, n_repeat)
n_adn_brian		= 4
n_pos_brian 	= 4
tcurves_brian 	= get_tuning_curves(n_adn_brian, np.linspace(0, 2*np.pi+0.01, 60))
tcurves_brian.columns = ['ADn.'+str(n) for n in range(n_adn_brian)]
tcurves_cla 	= get_tuning_curves(n_pos_brian, np.linspace(0, 2*np.pi+0.01, 60))/2.
tcurves_cla.columns = ['Pos.'+str(n) for n in range(n_pos_brian)]
from brian2 import *
start_scope()
freq_steps_adn 	= tcurves_brian.reindex(nang, method = 'nearest').values
freq_steps_pos 	= tcurves_cla.reindex(nang, method = 'nearest').values
stimadn 		= TimedArray(freq_steps_adn * Hz, dt = float(bin_size) * ms)
stimpos 		= TimedArray(freq_steps_pos * Hz, dt = float(bin_size) * ms)
eqs_pos 	= '''
dv/dt = -v / tau : 1
tau : second
'''
A   			= PoissonGroup(n_adn_brian, rates='stimadn(t, i)')
C 				= PoissonGroup(n_pos_brian, rates='stimpos(t, i)')
G 				= NeuronGroup(n_pos_brian, model=eqs_pos, threshold='v > 1', reset='v = 0', refractory=0*ms)
G.tau			= 50* ms
S1 				= Synapses(C, G, 'w : 1', on_pre='v_post += w')
S1.connect(i = np.arange(n_pos_brian), j = np.arange(n_pos_brian))
S1.w 			= 0.9
S2 				= Synapses(A, G, 'w : 1', on_pre='v_post += w')
S2.connect()
S2.delay 		= 0 * ms
weights 		= np.vstack(tcurves_brian.idxmax().values) - tcurves_cla.idxmax().values
weights2 		= 0.1*np.exp(1*np.cos(weights))/np.exp(1)
weights 		= np.exp(np.cos(weights))
S2.w 			= weights2.flatten()
#S2.w 			= 0.05
M 				= StateMonitor(G, 'v', record=True)
out_mon 		= SpikeMonitor(G)
inp_mon 		= SpikeMonitor(A)
pos_mon 		= SpikeMonitor(C)
duration 		= len(nang)*bin_size*ms
run(duration, report = 'text')



#####################################################################
# PLOTTING
#####################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale   # width in inches
	fig_height = fig_width*golden_mean*1.8             # height in inches
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




fig = figure(figsize = figsize(1))


matplotlib.rcParams.update({"axes.labelsize": 	6,
							"font.size": 		8,
							"legend.fontsize": 	8,
							"xtick.labelsize": 	5,
							"ytick.labelsize": 	5,   
							})               # Make the legend/label fonts a little smaller


##############################################################################################
# NETWORKS
##############################################################################################
gs1 	= gridspec.GridSpec(1,1)
gs1.update(hspace = 0.4, bottom = 0.69, top = 1, right = 0.5, left = 0.04)
ax 		= subplot(gs1[0,0])


for i, j in zip(S2.i, S2.j):    
	w = np.array(weights.flatten())
	plot([0, 1], [i, j], '-k', linewidth = w[i*n_pos_brian + j]*0.8)
for i, j in zip(S1.i, S1.j):    	
	plot([1.7, 1], [i, j], '-k', linewidth = 3.0)
plot(np.zeros(n_adn_brian), np.arange(n_adn_brian), 'ob', ms=10, markeredgecolor = 'blue')
plot(np.ones(n_pos_brian), np.arange(n_pos_brian), 'or', ms=10, markeredgecolor = 'red')
plot(np.ones(n_pos_brian)*1.7, np.arange(n_pos_brian), 'ob', ms=10, markeredgecolor = 'blue')


# xticks([0, 1, 1.5], ['f(ADn)', 'Pos', 'f(Pos)'])
xticks([])
yticks([])
noaxis(ax)
xlim(-0.6, 2.3)
ylim(-0.8, 3.5)

for n in range(n_adn_brian):
	sax = add_subplot_axes(ax,[0.0, n/float(n_adn_brian + 0.4) + 0.12, 0.15, 0.15])
	plot(tcurves_brian.iloc[:,n])
	sax.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
	sax.set_xticklabels(['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
	sax.set_yticks([])

for n in range(n_pos_brian):
	sax = add_subplot_axes(ax,[0.85, n/(float(n_adn_brian)+0.4) + 0.12, 0.15, 0.15])
	plot(tcurves_cla.iloc[:,n])
	sax.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
	sax.set_xticklabels(['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
	sax.set_yticks([])


ax.text(-0.2, -0.6, "T(ADn)")
ax.text(0.8, -0.6, "PoSub")
ax.text(1.45, -0.6, "T(PoSub)")
ax.annotate('', xytext=(0.3, -0.5), xy=(0.8, -0.5),
             arrowprops=dict(arrowstyle="->"))
ax.annotate('', xytext=(1.4, -0.5), xy=(1.18, -0.5),
             arrowprops=dict(arrowstyle="->"))

# gcf().text(0.02, 0.8, "Cross-Correlation (ADn / PoSub)", fontsize=13)


##############################################################################################
# SIMULATION
##############################################################################################
gs1 	= gridspec.GridSpec(5,1)
gs1.update(hspace = 0.2, bottom = 0.36, top = 0.66, right = 0.5, left = 0.04)


# ANGLE
ax 		= subplot(gs1[0,0])
ang 	= pd.DataFrame(index = np.arange(0, len(ang)*0.025, 0.025), data = ang)
plot(ang, color = 'grey')
yticks(np.arange(0, 2*np.pi, np.pi/4), ['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
# ylabel("Ang (rad)")
ylim(0, 2*np.pi)
xticks([])
ax.text(0.3, 3*np.pi/2 - np.pi/4., "Ang(rad)")
simpleaxis(ax)
ax.spines['bottom'].set_visible(False)

# ADN
ax 		= subplot(gs1[1,0])
plot(inp_mon.t, inp_mon.i, '|k', markersize = 4, color = 'blue')
simpleaxis(ax)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
yticks(np.arange(4), np.arange(1,5))
ax.set_ylim(-0.5, 3.5)
# ax.text(0.3, 3, "T(ADn)")
ylabel("T(ADn)")


ax 		= subplot(gs1[2,0])
plot(pos_mon.t, pos_mon.i, '|k', markersize = 4, color = 'blue')
simpleaxis(ax)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
yticks(np.arange(4), np.arange(1,5))
ax.set_ylim(-0.5, 3.5)
# ax.text(0.3, 3, "T(PoSub)")
ylabel("T(PoSub)")

# POS
ax 		= subplot(gs1[3:,0])
for i in np.arange(n_pos_brian):
	plot(M.t, M.v[i]+2*i, color = 'red', linewidth = 0.6)
plot(pos_mon.t, pos_mon.i*2+1.4, '|r', markersize = 4)	
xlabel("Time (s)", labelpad = 1)
simpleaxis(ax)
ylim(-0.2,)
yticks([0, 2, 4, 6], np.arange(1, 5))
# ax.text(1, 8.2, "PoSub")
xticks(np.arange(0, 16, 2))
ylabel("PoSub")

##############################################################################################
# INDIVIDUAL CROSS_CORR
##############################################################################################
n_pos_brian = 10
n_adn_brian = 10
gs1 = gridspec.GridSpec(n_pos_brian, n_adn_brian)
gs1.update(hspace = 0.1,wspace = 0.1, bottom = 0.0, top = 0.3, right = 0.5, left = 0.04)

store_cc 	= pd.HDFStore("../data/fig6_corr_simu_examples_10n.h5")
corr_simu 	= store_cc['data'] 
tcurves 	= store_cc['tcurves']
store_cc.close()
z1 = corr_simu

phi = np.vstack(tcurves.filter(regex='ADn.*').idxmax().values) - tcurves.filter(regex='Pos.*').idxmax().values
phi += 2*np.pi
phi %= 2*np.pi
rgb = phi / (2*np.pi)

from matplotlib.colors import hsv_to_rgb

for i in range(n_adn_brian):
	for j in range(n_pos_brian):
		ax = subplot(gs1[n_adn_brian-i-1,j])
		y = corr_simu[i*n_pos_brian + j]
		fill_between(y.index.values, np.zeros(len(y)), y.values, color = hsv_to_rgb([rgb[i,j], 1, 1]), alpha = 0.5)
		plot(y, '-', color = 'black')		
		# noaxis(ax)
		xlim(-500,500)
		ax.set_xticks([])
		ax.set_yticks([])		
		# ax.patch.set_facecolor(hsv_to_rgb([rgb[i,j], 1, 1]))
		if i == 9 and j == 4:			
			ax.text(0.5, 1.3, 'Cross-correlation T(ADn) / PoSub', fontsize = 9, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)
		
		if i == 0 and j == 4:
			ax.annotate('', xy=(-4, -0.3), xycoords='axes fraction', xytext=(6.2, -0.3), 
            	arrowprops=dict(arrowstyle="<->", color='black'))		
			ax.text(0.5, -0.8, 'PoSub (n = 10)', fontsize = 9, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)
		if i == 0 and j == 0:
			ax.text(0.5, -0.8, '0', fontsize = 9, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)			
		if i == 0 and j == 9:
			ax.text(0.5, -0.8, r'$2\pi$', fontsize = 9, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)			


		if i == 4 and j == 0:
			ax.annotate('', xy=(-0.3, -4), xycoords='axes fraction', xytext=(-0.3, 6.2), 
            	arrowprops=dict(arrowstyle="<->", color='black'))		
			ax.text(-0.6, 0.5, 'T(ADn) (n = 10)', fontsize = 9, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes, rotation = 90)
		if i == 0 and j == 0:
			ax.text(-0.8, 0.5, '0', fontsize = 9, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes, rotation = 90)			
		if i == 9 and j == 0:
			ax.text(-0.8, 0.5, r'$2\pi$', fontsize = 9, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes, rotation = 90)			



##############################################################################################
# CROSS CORR EXEMPLE
##############################################################################################
gs1 	= gridspec.GridSpec(3,2)
gs1.update(hspace = 0.3,wspace = 0.3, bottom = 0.36, top = 0.97, right = 1.0, left = 0.6)

store_cc 	= pd.HDFStore("../data/fig6_corr_simu_times_1_bs_10.h5")
corr_simu = store_cc['data']
store_cc.close()
corr_simu = corr_simu.replace(np.inf, np.nan)
corr_simu = corr_simu.dropna(axis = 1, how = 'any')
z1 = corr_simu
store_cc 	= pd.HDFStore("../data/fig6_corr_simu_times_4_bs_10.h5")
corr_simu = store_cc['data']
store_cc.close()
corr_simu = corr_simu.replace(np.inf, np.nan)
corr_simu = corr_simu.dropna(axis = 1, how = 'any')
z4 = corr_simu




def get_fwhm(z):
	# normalize z
	z = z - z.min()
	z = z / z.max()
	difference = z.max() - z.min()
	HM = difference/2 + z.min()
	pos_extremum = z.idxmax()
	nearest_above = (np.abs(z.loc[pos_extremum:] - HM)).argmin()
	nearest_below = (np.abs(z.loc[:pos_extremum] - HM)).argmin()
	return nearest_above - nearest_below

fwhm_xc = pd.DataFrame(index = [1,2,4,8], columns = ['fwhm'])
fwhm_xc.loc[1, 'fwhm'] = get_fwhm(z1.var(1))
fwhm_xc.loc[4, 'fwhm'] = get_fwhm(z4.var(1))



from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a*np.exp(-b*x) + c

lambdaa = pd.DataFrame(index = [1,2,4,8], columns = ['l'])
lambdaa_xgb = pd.DataFrame(index = [1,2,4,8], columns = ['l'])

ax = subplot(gs1[0,0:])

# BBOX
bbox = ax.get_position().bounds
ai = axes([bbox[0]+bbox[2]*0.1,bbox[1]+bbox[3]*0.7, 0.06, 0.05])
simpleaxis(ai)
ai.axvline(0, color = 'black', linewidth = 0.8)
ai.plot(z1.mean(1), linewidth = 3, color = 'green')
ai.fill_between(z1.index.values, z1.mean(1)-z1.std(1), z1.mean(1)+z1.std(1), alpha= 0.2, linewidth = 0.0, color = 'green')
ai.set_xticks([-500,0,500])
ai.set_yticks([])
# ai.set_title("ADn / PoSub", y = 0.99, fontsize = 7)
ai.set_xlabel("Time (ms)", labelpad = 1.2)
ai.set_title("cross-corr", fontsize = 6)
ai.set_ylabel('a.u.')
#####

ax.axvline(0, color = 'black')
simpleaxis(ax)
x = z1.var(1)#.loc[-250:250]
x = x - x.min()
x = x / x.max()
y = x.loc[0:]
#y += 0.000001
t = (y.index.values)*1e-3
popt, pcov = curve_fit(func, t, y.values)
lambdaa.loc[1,'l'] = popt[1]



ax.plot(x, '-', color= 'green', linewidth = 1.5, label = r'Ang $\times 1$')
ax.plot(y.index.values, func(t, popt[0], popt[1], popt[2]), color = 'black')

x = z4.var(1)#.loc[-250:250]
x = x - x.min()
x = x / x.max()
y = x.loc[-10:]
# y += 0.000001
t = (y.index.values+10)*1e-3
popt, pcov = curve_fit(func, t, y.values)
lambdaa.loc[4,'l'] = popt[1]


ax.plot(x, '--', color = 'green', linewidth = 1.5, label = r'Ang $\times 4$', dashes=(4, 2))
ax.plot(y.index.values, func(t, popt[0], popt[1], popt[2]), color = 'black')


ax.legend(frameon = False, fontsize = 6, handlelength=3)

ax.set_title(r"$\bar{\sigma}^2$(ADn/PoSub)", fontsize = 9, y = 0.99)

xlabel("Time (ms)", labelpad = 1.2)


#################################################################################################
# XGBOOST
#################################################################################################
ax = subplot(gs1[1,0:])
simpleaxis(ax)
axvline(0, color = 'black', linewidth = 0.8)

store_xgb 	= pd.HDFStore("../data/fig6_xgb_peaks_times_0_bs_10.h5")
xgb_peaks = store_xgb['data']
store_xgb.close()

peaks = pd.DataFrame(index = [1,2,4,8], columns = ['p'])
peaks.loc[1,'p'] = xgb_peaks.mean(1).idxmax()
fwhm_xgb = pd.DataFrame(index = [1,2,4,8], columns = ['fwhm'])
fwhm_xgb.loc[1, 'fwhm'] = get_fwhm(xgb_peaks.mean(1))

# x = xgb_peaks.mean(1)
# x = x - x.min()
# x = x / x.max()
# y = x.loc[0:]
# t = (y.index.values)*1e-3
# popt, pcov = curve_fit(func, t, y.values)
# lambdaa_xgb.loc[1,'l'] = popt[1]

plot(xgb_peaks.mean(1), color = 'grey', label = r'Ang $\times 1$')
plot([0], [xgb_peaks.mean(1).max()], '*', color = 'grey')

store_xgb 	= pd.HDFStore("../data/fig6_xgb_peaks_times_4_bs_10.h5")
xgb_peaks = store_xgb['data']
store_xgb.close()

plot(xgb_peaks.mean(1), '--', color = 'grey', label = r'Ang $\times 4$', dashes=(4, 2))
plot([0], [xgb_peaks.mean(1).max()], '*', color = 'grey', label = 'Peaks')
legend(frameon = False, fontsize = 6, handlelength=3)

xlabel("Time (ms)", labelpad = 1.2)
xticks([-300,0,300])
ylabel("Gain (a.u.)")
title("XGB", fontsize = 9)

##################################################################################################
# EXP FIT
##################################################################################################
ax = subplot(gs1[2,0])
simpleaxis(ax)
xticks(np.arange(4), [r'$\times '+str(i)+'$' for i in [1, 2, 4, 8]])
xlim(-0.5, 3.5)
# ylim(-15, 0)

# for p in [2,4,8]:
# 	store_xgb 	= pd.HDFStore("../data/fig6_xgb_peaks_times_"+str(p)+"_bs_10.h5")
# 	xgb_peaks = store_xgb['data']
# 	store_xgb.close()	
# 	peaks.loc[p, 'p'] = xgb_peaks.mean(1).idxmax()
# 	fwhm_xgb.loc[p, 'fwhm'] = get_fwhm(xgb_peaks.mean(1))
# 	x = xgb_peaks.mean(1)
# 	x = x - x.min()
# 	x = x / x.max()
# 	y = x.loc[0:]
# 	t = (y.index.values)*1e-3
# 	popt, pcov = curve_fit(func, t, y.values)
# 	lambdaa_xgb.loc[p,'l'] = popt[1]


for s in [2,8]:
	store_cc 	= pd.HDFStore("../data/fig6_corr_simu_times_"+str(s)+"_bs_10.h5")
	corr_simu = store_cc['data']
	store_cc.close()
	corr_simu = corr_simu.replace(np.inf, np.nan)
	corr_simu = corr_simu.dropna(axis = 1, how = 'any')
	z = corr_simu
	x = z.var(1)#.loc[-250:250]
	x = x - x.min()
	x = x / x.max()
	y = x.loc[-10:]
	# y += 0.000001
	t = (y.index.values+10)*1e-3
	popt, pcov = curve_fit(func, t, y.values)
	lambdaa.loc[s,'l'] = popt[1]
	fwhm_xc.loc[s, 'fwhm'] = get_fwhm(z.var(1))

plot(1/lambdaa.values, 'o-', label = r'$\bar{\sigma}^2(t) = N e^{-\frac{t}{\lambda}} + C$', color = 'green', markersize = 3)
# plot(1/lambdaa_xgb.values, '.-', color = 'grey', markersize = 3)
ylabel(r'$\lambda$')
legend(frameon = False, fontsize = 6, bbox_to_anchor=(1,0.94), loc="lower right")

##################################################################################################
# FWHM
##################################################################################################
ax2 = subplot(gs1[2,1])
simpleaxis(ax2)
for p in [2,4,8]:
	store_xgb 	= pd.HDFStore("../data/fig6_xgb_peaks_times_"+str(p)+"_bs_10.h5")
	xgb_peaks = store_xgb['data']
	store_xgb.close()	
	peaks.loc[p, 'p'] = xgb_peaks.mean(1).idxmax()
	fwhm_xgb.loc[p, 'fwhm'] = get_fwhm(xgb_peaks.mean(1))

# ax2.plot(np.arange(4), peaks.values, '*-', color = 'grey', label = 'XGB Peak Time')

ax2.plot(np.arange(4), fwhm_xc.values, '.-', color = 'green', label = 'cross-corr')

ax2.plot(np.arange(4), fwhm_xgb.values, '^-', color = 'grey', label = 'XGB')
ax2.set_ylabel("Time (ms)")
# ax2.set_yticks([0])
ax2.set_title("FWHM", fontsize = 9)
ax2.set_xlim(-0.5, 3.5)
ax2.spines['top'].set_visible(False)

ax2.set_xticks(np.arange(4))
ax2.set_xticklabels([r'$\times '+str(i)+'$' for i in [1, 2, 4, 8]])

legend(frameon = False, fontsize = 6, bbox_to_anchor=(1.2,0.75), loc="lower right")

####################################################################################################
# PEAKS XGBOOST
####################################################################################################
gs1 	= gridspec.GridSpec(1,1)
gs1.update(hspace = 0.7,wspace = 0.5, bottom = 0.0, top = 0.3, right = 1.0, left = 0.6)

ax = subplot(gs1[0,0])
simpleaxis(ax)

store_fig6 = pd.HDFStore("../data/fig6_delays_xgb_peaks.h5")

delays = []
xp = [0, -10, -25, -50, -80, -105]
count = 0
for t in [0.0, 10.0, 20.0, 50.0, 80.0, 100.0]:
	delay = t
	delays.append(-delay)
	y = store_fig6['/'+str(t)]
	y = y.loc[-120:20]	
	ax.plot(y.mean(1), label = str(delay), color = 'grey')
	ax.fill_between(y.index.values, y.mean(1) - y.sem(1), y.mean(1) + y.sem(1), color = 'grey', alpha = 0.2, linewidth = 0.0)

	# text(x = float(y.mean(1).idxmax), y = y.mean(1).max(), 'yo')
	if t == 100.0:
		ax.text(xp[count]-10, 10000-count*500, r'$\delta=-$'+str(int(t))+r'$ms$', fontsize = 6, rotation = 0)
	else:
		ax.text(xp[count]-10, 10000-count*500, r'$-$'+str(int(t))+r'$ms$', fontsize = 6, rotation = 0)
	count += 1

ax.set_title(r'$T(ADn) \Rightarrow_{\delta}  PoSub$')
store_fig6.close()

xtk = delays
xtkl = [r'-' + str(int(np.abs(t))) for t in delays]
ax.set_xlim(-120,20)
ax.set_xticks(xtk)
ax.set_xticklabels(xtkl)
grid(axis = 'x')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Gain (a.u.)")

savefig("../../figures/fig6.pdf", dpi=900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig6.pdf &")



# tmp = pd.DataFrame(columns = [1,2,4,8])
# for s in [1,2,4,8]:
# 	store_cc 	= pd.HDFStore("../data/fig6_corr_simu_times_"+str(s)+"_bs_10.h5")
# 	corr_simu = store_cc['data']
# 	store_cc.close()
# 	corr_simu = corr_simu.replace(np.inf, np.nan)
# 	corr_simu = corr_simu.dropna(axis = 1, how = 'any')
# 	z = corr_simu
# 	x = z.var(1)#.loc[-250:250]	
# 	x = x - x.min()
# 	x = x / x.max()
# 	tmp[s] = x
# 	# y = x.loc[-10:]
# 	# # y += 0.000001
# 	# t = (y.index.values+10)*1e-3
# 	# popt, pcov = curve_fit(func, t, y.values)

# for s in tmp.columns:
# 	y = tmp[s]
# 	popt, pcov = curve_fit(func, t, y.values)
# 	print popt[1]
# 	plot(y.index.values, func(t, popt[0], popt[1], popt[2]))	
