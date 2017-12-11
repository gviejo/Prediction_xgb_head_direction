import scipy.io
import sys,os
import numpy as np
from matplotlib.pyplot import *
import pandas as pd

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



target =  sys.argv[-3]
xgb_bin_size = int(sys.argv[-2])
xgb_bin_length = int(sys.argv[-1])

real_bin_size = 5
bin_size = 1.25

# target = 'Pos.0'
# xgb_bin_size = 10
# xgb_bin_length = 500


store_data = pd.HDFStore("../data/fig6_frate_data_brian_delays_speed_"+str(int(real_bin_size/bin_size))+".h5")
data = store_data['data']
data = data.iloc[0:len(data)/2]
store_data.close()

adn_neuron = np.array([n for n in data.keys() if 'ADn' in n])
###########################################################################################
# SHIFTED BINS
###########################################################################################
nb_bins = xgb_bin_length/xgb_bin_size
time = np.arange(-(xgb_bin_length/2), (xgb_bin_length/2)+xgb_bin_size, xgb_bin_size)
duration = len(data)			

time_shifted = np.zeros(( duration-nb_bins, len(adn_neuron), nb_bins+1))
for n,i in zip(adn_neuron, range(len(adn_neuron))):	
	tmp = data[n]
	for j in range(0,nb_bins+1):
		# time_shifted[:,i,j] = tmp[40-j:duration-j]
		time_shifted[:,i,j] = tmp[j:duration-nb_bins+j]

time_shifted = time_shifted.reshape(time_shifted.shape[0],time_shifted.shape[1]*time_shifted.shape[2])	


#####################################################################
# LEARNING XGB
#####################################################################
import xgboost as xgb


params = {'objective': "count:poisson", #for poisson output
    'eval_metric': "logloss", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.1,
    'min_child_weight': 2, 'n_estimators': 580,
    'subsample': 0.6, 'max_depth': 5, 'gamma': 0.4}        
num_round = 90


Yall = data[target].values		
# need to cut Yall
Yall = Yall[nb_bins//2:-nb_bins//2]
data = None
print(time_shifted.shape)
print(Yall.shape)
dtrain = xgb.DMatrix(time_shifted, label=Yall)
bst = xgb.train(params, dtrain, num_round)

#####################################################################
# EXTRACT TREE STRUCTURE
#####################################################################
threshold = extract_tree_threshold(bst)

#####################################################################
# EXTRACT GAIN VALUE
#####################################################################
gain = bst.get_score(importance_type = 'gain')

#####################################################################
# CONVERT TO TIMING OF SPLIT POSITION
#####################################################################
time_count = np.zeros((len(adn_neuron), nb_bins+1))
index = np.repeat(np.arange(len(adn_neuron)), nb_bins+1)

splits = threshold
for s in splits.keys():
	time_count[index[int(s[1:])], int(s[1:])%(nb_bins+1)] = len(splits[s])

time_count = time_count.sum(0)

gain_value = np.zeros((len(adn_neuron), nb_bins+1))
for s in gain.keys():
	gain_value[index[int(s[1:])], int(s[1:])%(nb_bins+1)] = gain[s]

# gain_value = gain_value.reshape(len(pos_neuron)*len(adn_neuron), 41)
gain_value = gain_value.sum(0)


time = np.arange(-(xgb_bin_length/2), (xgb_bin_length/2)+xgb_bin_size, xgb_bin_size)
xgb_peaks = pd.DataFrame(index = time, data = (time_count*gain_value).transpose())


store_xgb = pd.HDFStore("../data/fig_6_xgb_output_brian_delays_speed_"+str(int(real_bin_size/bin_size))+".h5")
store_xgb[target] = xgb_peaks
store_xgb.close()
