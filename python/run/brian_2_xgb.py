import scipy.io
import sys,os
import numpy as np
from matplotlib.pyplot import *
import pandas as pd
import xgboost as xgb

store 			= pd.HDFStore("../data/spikes_brian.h5")
data 			= store['data']
store.close()

adn_neuron = [n for n in data.keys() if 'ADn' in n]
pos_neuron = [n for n in data.keys() if 'Pos' in n]

bin_size = 25
bin_length = 500

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



###########################################################################################
# create shifted spiking activity from -500 to 500 ms with index 0 to 40 (20 for t = 0 ms) for all ADn_neuron
# remove 20 points at the beginning and 20 points at the end
###########################################################################################
nb_bins = bin_length/bin_size
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

time_shifted = time_shifted.reshape(time_shifted.shape[0],time_shifted.shape[1]*time_shifted.shape[2])	



for k in combination.keys():
	print(k)
	features = combination[k]['features']
	targets = combination[k]['targets']					
	Yall = data[targets].values		
	# need to cut Yall
	Yall = Yall[nb_bins//2:-nb_bins//2]
	print(time_shifted.shape)
	print(Yall.shape)
	dtrain = xgb.DMatrix(time_shifted, label=Yall)
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


time = np.arange(-(bin_length/2), (bin_length/2)+bin_size, bin_size)


xgb_peaks = pd.DataFrame(index = time, data = (time_count*gain_value).transpose())




#####################################################################
# PLOT
#####################################################################

plot(time, xgb_peaks.mean(1))
title("XGB")


show()


