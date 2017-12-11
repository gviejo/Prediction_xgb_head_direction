import scipy.io
import sys,os
import numpy as np
from matplotlib.pyplot import *
import pandas as pd
from scipy.ndimage import gaussian_filter1d as gfilt

###########################################################################
# BUILD TUNING CURVE
###########################################################################
def tuning_curve(x, f, nb_bins, tau = 40.0):	
	bins = np.linspace(0, 2*np.pi, nb_bins+1)
	index = np.digitize(x, bins).flatten()    
	tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)]).astype('float')  	
	occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)]).astype('float')
	
	tcurve = (tcurve/occupancy)*tau	
	x = bins[0:-1] + (bins[1]-bins[0])/2.
	
	return (x, tcurve, index-1)


real_bin_size = 5
bin_size = 5

matdata = scipy.io.loadmat("../data/sessions_nosmoothing_"+str(real_bin_size)+"ms/wake/boosted_tree.Mouse28-140313.mat")
n_adn 	= matdata['ADn'].shape[1]
n_pos 	= matdata['Pos'].shape[1]
ang 	= matdata['Ang'].flatten()
tcurves = []
columns = []
for k, n_neurons in zip(['ADn', 'Pos'],[n_adn, n_pos]):	
	for n in range(n_neurons):
		columns.append(k+"."+str(n))
		xbins, frate, index = tuning_curve(ang, matdata[k][:,n], 60, 1/(real_bin_size*1e-3))
		tcurves.append(frate)

tcurves = pd.DataFrame(index = xbins, columns = columns, data = np.array(tcurves).transpose())
adn_neuron = ["ADn."+str(i) for i in np.arange(n_adn)]

############################################################################
# SIMULATION
############################################################################
def get_tuning_curves(n, x):
	# phi = np.sort(np.random.uniform(0, 2*np.pi, n))
	phi = np.linspace(0, 2*np.pi, n)
	# A 	= np.random.uniform(0, 4, n)
	# B 	= np.random.uniform(0.00001, 0.00007, n)
	# K 	= np.random.uniform(12, 14, n)
	# return pd.DataFrame(index = x, columns = np.arange(n), data = A+B*np.exp(K*np.cos(np.vstack(x) - phi)))	
	A 	= np.random.uniform(10, 50,n)
	B 	= np.random.uniform(5, 10, n)
	C 	= np.random.uniform(0, 2, n)
	return pd.DataFrame(index = x, columns = np.arange(n), data = C+A*np.exp(B*np.cos(np.vstack(x) - phi))/np.exp(B))



# ang = np.load("../data/ang_sws_sample.npy")[0:5000]


n_repeat 		= 5
nang 			= np.tile(ang, n_repeat)

n_adn_brian		= 10
n_pos_brian 	= 10
tcurves_brian 	= get_tuning_curves(n_adn_brian, tcurves.index.values)
tcurves_brian.columns = ['ADn.'+str(n) for n in range(n_adn_brian)]
tcurves_cla 	= get_tuning_curves(n_pos_brian, tcurves.index.values)/2.
tcurves_cla.columns = ['Pos.'+str(n) for n in range(n_pos_brian)]


from brian2 import *
set_device('cpp_standalone')
start_scope()

freq_steps_adn 	= tcurves_brian.reindex(nang, method = 'nearest').values
freq_steps_pos 	= tcurves_cla.reindex(nang, method = 'nearest').values
# freq_steps_pos 	= gfilt(freq_steps_pos, 10, axis = 0)
# freq_steps_pos 	= freq_steps_pos - np.min(freq_steps_pos, axis = 0)
# freq_steps_pos 	= (freq_steps_pos / (np.max(freq_steps_pos, axis = 0)))*1.1
stimadn 		= TimedArray(freq_steps_adn * Hz, dt = float(bin_size) * ms)
stimpos 		= TimedArray(freq_steps_pos * Hz, dt = float(bin_size) * ms)


sigma 			= 0.0
# eqs_neurons = '''
# dv/dt = -v/tau + sigma*xi*tau**-0.5 : volt
# I : 1
# tau : second
# '''
eqs_pos 	= '''
dv/dt = -v / tau + sigma * (2 / tau)**.5 * xi : 1
tau : second
'''
# eqs_pos 	= 	'''
# dv/dt = -v / tau : 1
# tau : second
# '''
# eqs_pos = '''
# dv/dt = (-v + stimpos(t,i))/(tau) : 1
# tau : second
# '''
A   			= PoissonGroup(n_adn_brian, rates='stimadn(t, i)')
C 				= PoissonGroup(n_pos_brian, rates='stimpos(t, i)')
G 				= NeuronGroup(n_pos_brian, model=eqs_pos, threshold='v > 1', reset='v = 0', refractory=0*ms)
G.tau			= 50* ms
S1 				= Synapses(C, G, 'w : 1', on_pre='v_post += w')
S1.connect(i = np.arange(n_pos_brian), j = np.arange(n_pos_brian))
# S1.connect(p = 0.2)
S1.w 			= 0.9
S2 				= Synapses(A, G, 'w : 1', on_pre='v_post += w')
S2.connect()
S2.delay 		= 0 * ms
weights 		= np.vstack(tcurves_brian.idxmax().values) - tcurves_cla.idxmax().values
weights 		= 0.1*np.exp(10*np.cos(weights))/np.exp(10)
S2.w 			= weights.flatten()
#S2.w 			= 0.05
# M 				= StateMonitor(G, 'v', record=True)
out_mon 		= SpikeMonitor(G)
inp_mon 		= SpikeMonitor(A)

duration 		= (len(nang))*bin_size*ms
run(duration, report = 'text')

###########################################################################
# TUNING CURVES OF POS NEURONS
###########################################################################
inp_spikes 		= inp_mon.spike_trains()
out_spikes 		= out_mon.spike_trains()
for k in out_spikes.keys():
	f, bins_edge = np.histogram(out_spikes[k]/ms, int(duration / ms / bin_size), range = (0, duration/ms))
	x, tcurve, index = tuning_curve(nang, f, 60, 1/float(bin_size*1e-3))
	tcurves_brian['Pos.'+str(k)] = pd.DataFrame(data = tcurve, index = x)

for k in out_spikes.keys():
	f, bins_edge = np.histogram(inp_spikes[k]/ms, int(duration / ms / bin_size), range = (0, duration/ms))
	x, tcurve, index = tuning_curve(nang, f, 60, 1/float(bin_size*1e-3))
	tcurves_cla['ADn.'+str(k)] = pd.DataFrame(data = tcurve, index = x)


#####################################################################
# CROSS_CORRELATION OF SIMULATED NEURONS
#####################################################################
def crossCorr(t1, t2, binsize, nbins):
	''' 
		Fast crossCorr 
	'''
	nt1 = len(t1)
	nt2 = len(t2)
	if np.floor(nbins/2)*2 == nbins:
		nbins = nbins+1

	m = -binsize*((nbins+1)/2)
	B = np.zeros(nbins)
	for j in range(nbins):
		B[j] = m+j*binsize

	w = ((nbins/2) * binsize)
	C = np.zeros(nbins)
	i2 = 1

	for i1 in range(nt1):
		lbound = t1[i1] - w
		while i2 < nt2 and t2[i2] < lbound:
			i2 = i2+1
		while i2 > 1 and t2[i2-1] > lbound:
			i2 = i2-1

		rbound = lbound
		l = i2
		for j in range(nbins):
			k = 0
			rbound = rbound+binsize
			while l < nt2 and t2[l] < rbound:
				l = l+1
				k = k+1

			C[j] += k

	# for j in range(nbins):
	# C[j] = C[j] / (nt1 * binsize)
	C = C/(nt1 * binsize/1000)

	return C


cc_bin_size = 25
cc_bin_length = 1000
time_cc = np.arange(-(cc_bin_length/2), (cc_bin_length/2)+cc_bin_size, cc_bin_size)

corr_simu = pd.DataFrame(index = time_cc, data = [], columns = [])
for i in np.arange(n_adn_brian):
	for j in np.arange(n_pos_brian):
		C = crossCorr(np.round(out_spikes[j]/ms), np.round(inp_spikes[i]/ms), cc_bin_size, cc_bin_length/cc_bin_size)		
		mean_firing_rate = float(len(inp_spikes[i]))/float(duration/second)
		corr_simu[j+i*n_pos_brian] = C/mean_firing_rate



store_cc 	= pd.HDFStore("../data/fig6_corr_simu_examples_10n.h5")
store_cc['data'] = corr_simu
store_cc['tcurves'] = tcurves_brian
store_cc.close()
sys.exit()

print("Cross corr finished")

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k', linewidth = S.w[i*n_pos_brian + j]*60.0)
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    scatter(S.i, S.j, s = S.w*10.0, c = 'k')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')


# visualise_connectivity(S2)

# count = 1
# figure()
# for i in np.arange(n_adn_brian):
# 	for j in np.arange(n_pos_brian):
# 		subplot(n_adn_brian, n_pos_brian, count)
# 		plot(corr_simu[j+i*n_pos_brian])
# 		axvline(0)
# 		xticks([])
# 		yticks([])
# 		count += 1

# figure()
# subplot(221)
# plot(tcurves_brian.filter(regex='ADn.*'))
# plot(tcurves_cla.filter(regex='ADn.*'))
# subplot(223)
# plot(tcurves_brian.filter(regex='Pos.*'))
# # plot(tcurves_cla.filter(regex='Pos.*'))
# subplot(222)

# corr_simu = corr_simu.replace(np.inf, np.nan)
# corr_simu = corr_simu.dropna(axis = 1, how = 'any')
# # mean centered
# # z = corr_simu - corr_simu.mean(0)
# # z = z / z.std(0)
# z = corr_simu
# plot(z.mean(1), label = 'simu', linewidth = 3)
# fill_between(z.index.values, z.mean(1)-z.std(1), z.mean(1)+z.std(1), alpha= 0.5)

# subplot(224)
# plot(z.var(1))

# show()



##########################################################################
# DATA FOR XGBOOST
##########################################################################
# import xgboost as xgb
# import pandas as pd

# def extract_tree_threshold(trees):
# 	n = len(trees.get_dump())
# 	thr = {}
# 	for t in xrange(n):
# 		gv = xgb.to_graphviz(trees, num_trees=t)
# 		body = gv.body		
# 		for i in xrange(len(body)):
# 			for l in body[i].split('"'):
# 				if 'f' in l and '<' in l:
# 					tmp = l.split("<")
# 					if thr.has_key(tmp[0]):
# 						thr[tmp[0]].append(float(tmp[1]))
# 					else:
# 						thr[tmp[0]] = [float(tmp[1])]					
# 	for k in thr.iterkeys():
# 		thr[k] = np.sort(np.array(thr[k]))
# 	return thr

xgb_bin_size = 10 # ms
xgb_bin_length = 500
data = pd.DataFrame()

for k in inp_spikes.keys():
	f, bins_edge = np.histogram(inp_spikes[k]/ms, int(duration / ms / xgb_bin_size), range = (0, duration/ms))
	data['ADn'+'.'+str(k)] = f

for k in out_spikes.keys():
	f, bins_edge = np.histogram(out_spikes[k]/ms, int(duration / ms / xgb_bin_size), range = (0, duration/ms))
	data['Pos'+'.'+str(k)] = f




# store 			= pd.HDFStore("../data/spikes_brian.h5")
# store['data'] 	= data
# store.close()

# sys.exit()


data = data.iloc[0:len(data)/2]

adn_neuron = [n for n in data.keys() if 'ADn' in n]
pos_neuron = [n for n in data.keys() if 'Pos' in n]




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
nb_bins = xgb_bin_length/xgb_bin_size
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
import xgboost as xgb
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


time = np.arange(-(xgb_bin_length/2), (xgb_bin_length/2)+xgb_bin_size, xgb_bin_size)


xgb_peaks = pd.DataFrame(index = time, data = (time_count*gain_value).transpose())

store_xgb 	= pd.HDFStore("../data/fig6_xgb_peaks_times_"+str(int(real_bin_size/bin_size))+"_bs_"+str(xgb_bin_size)+".h5")
store_xgb['data'] = xgb_peaks
store_xgb.close()


#####################################################################
# PLOT
#####################################################################

plot(time, xgb_peaks.mean(1))
title("XGB")


show()


