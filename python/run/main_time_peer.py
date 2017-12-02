#!/usr/bin/env python

'''
    File name: main_time_peer.py
    Author: Guillaume Viejo
    Date created: 04/07/2017    
    Python Version: 2.7

To compute the split time position for figure 4

'''

import warnings
import pandas as pd
import scipy.io
import numpy as np
# Should not import fonctions if already using tensorflow for something else
# from fonctions import *
import sys, os
import itertools
import cPickle as pickle
import xgboost as xgb

import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib.pyplot import *

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
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


datatosave = {}

episode = ['wake', 'rem', 'sws']
# episode = ['sws']
bin_size = 5 # ms

for ep in episode:

	sessions = os.listdir("../data/sessions_nosmoothing_"+str(bin_size)+"ms/"+ep)
	datatosave[ep] = []

	for sess in sessions:

		print(ep, sess)

		#####################################################################
		# DATA LOADING
		#####################################################################
		adrien_data = scipy.io.loadmat("../data/sessions_nosmoothing_"+str(bin_size)+"ms/"+ep+"/"+sess)

		#####################################################################
		# DATA ENGINEERING
		#####################################################################
		data 			= 	pd.DataFrame()
		data['time'] 	= 	np.arange(len(adrien_data['Ang']))		# TODO : import real time from matlab script		
		# Firing data
		for i in xrange(adrien_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = adrien_data['Pos'][:,i]
		for i in xrange(adrien_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = adrien_data['ADn'][:,i]

		# cut if longer than 40 min
		if len(data) > (15*60*1000)//bin_size:
			data = data[0:(40*60*1000)//bin_size]


		#####################################################################
		# COMBINATIONS DEFINITION
		#####################################################################
		adn_neuron = [n for n in data.keys() if 'ADn' in n]
		pos_neuron = [n for n in data.keys() if 'Pos' in n]

		if len(adn_neuron) >= 7:

			# create shifted spiking activity from -500 to 500 ms with index 0 to 40 (20 for t = 0 ms) for all ADn_neuron
			# remove 20 points at the beginning and 20 points at the end
			nb_bins = 1000/bin_size
			duration = len(data)			
			time_shifted = np.zeros(( duration-nb_bins, len(adn_neuron), nb_bins+1))
			for n,i in zip(adn_neuron, range(len(adn_neuron))):	
				tmp = data[n]
				for j in range(0,nb_bins+1):
					# time_shifted[:,i,j] = tmp[40-j:duration-j]
					time_shifted[:,i,j] = tmp[j:duration-nb_bins+j]


			combination = {}
			for k in pos_neuron:
				combination[k] = {	'features'	: adn_neuron,
									'targets'	: k,						
									}

			#####################################################################
			# LEARNING XGB
			#####################################################################

			bsts = {i:{} for i in combination.iterkeys()} # to keep the boosted tree
			params = {'objective': "count:poisson", #for poisson output
			    'eval_metric': "logloss", #loglikelihood loss
			    'seed': 2925, #for reproducibility
			    'silent': 1,
			    'learning_rate': 0.1,
			    'min_child_weight': 2, 'n_estimators': 580,
			    'subsample': 0.6, 'max_depth': 5, 'gamma': 0.4}        
			num_round = 90

			for k in combination.keys():
				features = combination[k]['features']
				targets = combination[k]['targets']				
				X = time_shifted.reshape(time_shifted.shape[0],time_shifted.shape[1]*time_shifted.shape[2])	
				Yall = data[targets].values		
				# need to cut Yall
				Yall = Yall[nb_bins//2:-nb_bins//2]
				dtrain = xgb.DMatrix(X, label=Yall)
				bst = xgb.train(params, dtrain, num_round)
				bsts[k] = bst

			#####################################################################
			# EXTRACT TREE STRUCTURE
			#####################################################################
			thresholds = {}
			for i in bsts.iterkeys():
				thresholds[i] = extract_tree_threshold(bsts[i])		

			#####################################################################
			# EXTRACT GAIN VALUE
			#####################################################################
			gain = {}
			for i in bsts.iterkeys():
				gain[i] = bsts[i].get_score(importance_type = 'gain')
			
			#####################################################################
			# CONVERT TO TIMING OF SPLIT POSITION
			#####################################################################
			time_count = np.zeros((len(pos_neuron), len(adn_neuron), nb_bins+1))
			index = np.repeat(np.arange(len(adn_neuron)), nb_bins+1)
			for n in thresholds.iterkeys():
				splits = thresholds[n]
				for s in splits.keys():
					time_count[int(n.split(".")[1]), index[int(s[1:])], int(s[1:])%(nb_bins+1)] = len(splits[s])

			time_count = time_count.sum(1)

			gain_value = np.zeros((len(pos_neuron), len(adn_neuron), nb_bins+1))
			index = np.repeat(np.arange(len(adn_neuron)), nb_bins+1)
			for n in gain.iterkeys():
				g = gain[n]
				for s in g.keys():
					gain_value[int(n.split(".")[1]), index[int(s[1:])], int(s[1:])%(nb_bins+1)] = g[s]

			# gain_value = gain_value.reshape(len(pos_neuron)*len(adn_neuron), 41)
			gain_value = gain_value.sum(1)

			
			datatosave[ep].append(time_count*gain_value)
			# datatosave[ep].append(gain_value)

			

	datatosave[ep] = np.vstack(datatosave[ep])


	with open("../data/fig4_time_peer_"+ep+"_5ms_bins_guillaume.pickle", 'wb') as f:
		pickle.dump(datatosave[ep], f)
