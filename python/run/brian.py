import scipy.io
import sys,os
import numpy as np
from matplotlib.pyplot import *
import pandas as pd

###########################################################################
# BUILD TUNING CURVE
###########################################################################
def tuning_curve(x, f, nb_bins, tau = 40.0):	
	bins = np.linspace(x.min(), x.max()+1e-8, nb_bins+1)
	index = np.digitize(x, bins).flatten()    
	tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)]).astype('float')  	
	occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)]).astype('float')
	
	tcurve = (tcurve/occupancy)*tau	
	x = bins[0:-1] + (bins[1]-bins[0])/2.
	
	return (x, tcurve, index-1)

# sessions = os.listdir("../data/sessions_nosmoothing_25ms/wake/")
# for s in sessions:
# 	matdata = scipy.io.loadmat("../data/sessions_nosmoothing_25ms/wake/"+s)
# 	print(s, matdata['ADn'].shape[1], matdata['Pos'].shape[1])	


bin_size = 5

matdata = scipy.io.loadmat("../data/sessions_nosmoothing_"+str(bin_size)+"ms/wake/boosted_tree.Mouse28-140313.mat")
n_adn 	= matdata['ADn'].shape[1]
n_pos 	= matdata['Pos'].shape[1]
ang 	= matdata['Ang'].flatten()
tcurves = []
columns = []
for k, n_neurons in zip(['ADn', 'Pos'],[n_adn, n_pos]):	
	for n in range(n_neurons):
		columns.append(k+"."+str(n))
		xbins, frate, index = tuning_curve(ang, matdata[k][:,n], 60, 1/(bin_size*1e-3))
		tcurves.append(frate)

tcurves = pd.DataFrame(index = xbins, columns = columns, data = np.array(tcurves).transpose())

###########################################################################
# CROSS-CORR OF REAL NEURONS
###########################################################################
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

# spikes = scipy.io.loadmat("../data/spikes.Mouse28-140313.mat")
# spikes_ts = dict()
# for k, n_neurons in zip(['ADn', 'Pos'],[n_adn, n_pos]):	
# 	tmp = spikes[k][0][0][0]
# 	for n in range(n_neurons):
# 		spikes_ts[k+"."+str(n)] = tmp[n][0][0,0][1][0][0][2].flatten()
		
matdata2 		= scipy.io.loadmat("../data/spikes_binned.Mouse28-140313.mat")
spikes_ts = dict()
for k in ['ADn', 'Pos']:
	for n in range(matdata2[k].shape[1]):
		tmp 					= matdata2[k][:,n]
		spikes_ts[k+"."+str(n)] = np.arange(0, len(matdata2[k])*0.001, 0.001)[tmp == 1.0]



# # TO SHOW ALL CROSS CORR OF ALL PAIRS OF POS_ADN NEURONS

# corr_real = []
# real_adn_neuron = ["ADn."+str(i) for i in np.arange(n_adn)]
# real_pos_neuron = ["Pos."+str(j) for j in np.arange(n_pos)]
# count = 1
# figure()
# for m in real_adn_neuron:
# 	for n in real_pos_neuron:		
# 		C = crossCorr(spikes_ts[n]*1000, spikes_ts[m]*1000, cc_bin_size, 1000/cc_bin_size)
# 		# frate = float(len(spikes_ts[m]))/float(spikes_ts[m][-1] - spikes_ts[m][0])
		
# 		subplot(n_adn, n_pos, count)
# 		plot(C)
# 		title(m+" "+n, fontsize = 9)
# 		xticks([])
# 		yticks([])
# 		axvline(20)
# 		count += 1
# 		corr_real.append(C)
# figure()
# count = 1
# for m in real_adn_neuron:
# 	for n in real_pos_neuron:
# 		subplot(n_adn, n_pos, count)
# 		plot(tcurves[m])
# 		plot(tcurves[n])
# 		title(m+" "+n, fontsize = 9)
# 		count += 1
# show()

cc_bin_size = 25
cc_bin_length = 1000
time_cc = np.arange(-(cc_bin_length/2), (cc_bin_length/2)+cc_bin_size, cc_bin_size)
to_keep = ['ADn.15', 'Pos.2']
# to_keep = ['ADn.4', 'Pos.2']
# to_keep = ['ADn.7', 'Pos.8']
# to_keep = ['ADn.11', 'Pos.1']
corr_real = crossCorr(spikes_ts[to_keep[1]]*1000, spikes_ts[to_keep[0]]*1000, cc_bin_size, cc_bin_length/cc_bin_size)
corr_real = pd.DataFrame(index = time_cc, data = corr_real, columns = ['real'])

adn = tcurves[to_keep[0]]
pos = tcurves[to_keep[1]]


############################################################################
# SIMULATION GRID SEARCH
############################################################################
from brian2 import *
set_device('cpp_standalone')
start_scope()


eqs_neurons 	= 	'''
					dv/dt = -v / tau : 1 
					tau : second
					'''

# repeating the same simulation several times to ensure good statistics
n_repeat 		= 1
nang 			= np.tile(ang, n_repeat)
# freq_steps_adn 	= adn.reindex(nang, method = 'nearest').values.flatten()
freq_steps_pos 	= pos.reindex(nang, method = 'nearest').values.flatten()
# false tuning curve for the false pos to receive input from adn
# dummy_tcurve 	= pd.DataFrame(	index 	= adn.index.values, 
								# data 	= 0.2*np.exp(3.0*np.cos(adn.index.values - adn.index.values[np.argmax(pos.values)])))
# freq_steps_pos 	= dummy_tcurve.reindex(nang, method = 'nearest').values.flatten()
I_adn 			= SpikeGeneratorGroup(1, np.zeros(len(spikes_ts[to_keep[0]])), times = spikes_ts[to_keep[0]]*second)
# stimadn 		= TimedArray(freq_steps_adn * Hz, dt = float(bin_size) * ms)
stimpos 		= TimedArray(freq_steps_pos * Hz, dt = float(bin_size) * ms)
# I_adn 			= PoissonGroup(1, rates='stimadn(t)')
I_pos 			= PoissonGroup(1, rates='stimpos(t)')

weights 		= np.arange(0, 1.0, 0.1) # ADn weights
# delays 			= [0.0, 50., 100.]*ms # ADn -> output
taus 			= np.arange(5, 200, 40)*ms

G 				= NeuronGroup(len(weights)*len(weights)*len(taus), model=eqs_neurons, threshold='v > 1.0', reset='v = 0', method = 'exact')
G.tau 			= np.tile(np.repeat(taus, len(weights)), len(weights))
S_adn 			= Synapses(I_adn, G, 'w : 1', on_pre='v_post += w')
S_adn.connect(i=0, j=np.arange(0,len(G)))
S_adn.w 		= np.tile(np.tile(weights, len(taus)), len(weights))
# S_adn.delay 	= np.repeat(delays, len(weights)*len(taus))
S_adn.delay 	= 0*ms
S_pos 			= Synapses(I_pos, G, 'w : 1', on_pre='v_post += w')
S_pos.connect(i=0, j=np.arange(0,len(G)))
S_pos.w 		= np.tile(np.tile(weights, len(taus)), len(weights))
out_mon 		= SpikeMonitor(G)
adn_mon 		= SpikeMonitor(I_adn)
pos_mon 		= SpikeMonitor(I_pos)
duration 		= len(nang)*bin_size*ms
run(duration, report = 'text')
print(" Simulation finished")
adn_spikes = adn_mon.spike_trains()[0]
pos_spikes = pos_mon.spike_trains()[0]
out_spikes = {n:out_mon.spike_trains()[n] for n in out_mon.spike_trains().keys()}

############################################################################
# SIMULATION
############################################################################
# from brian2 import *
# set_device('cpp_standalone')
# start_scope()


# eqs_neurons 	= 	'''
# 					dv/dt = -v / tau : 1 
# 					tau : second
# 					'''

# # repeating the same simulation several times to ensure good statistics
# n_repeat 		= 1
# nang 			= np.tile(ang, n_repeat)
# # freq_steps_adn 	= adn.reindex(nang, method = 'nearest').values.flatten()
# freq_steps_pos 	= pos.reindex(nang, method = 'nearest').values.flatten()
# # false tuning curve for the false pos to receive input from adn
# # dummy_tcurve 	= pd.DataFrame(	index 	= adn.index.values, 
# 								# data 	= 0.2*np.exp(3.0*np.cos(adn.index.values - adn.index.values[np.argmax(pos.values)])))
# # freq_steps_pos 	= dummy_tcurve.reindex(nang, method = 'nearest').values.flatten()
# I_adn 			= SpikeGeneratorGroup(1, np.zeros(len(spikes_ts[to_keep[0]])), times = spikes_ts[to_keep[0]]*second)
# # stimadn 		= TimedArray(freq_steps_adn * Hz, dt = float(bin_size) * ms)
# stimpos 		= TimedArray(freq_steps_pos * Hz, dt = float(bin_size) * ms)
# # I_adn 			= PoissonGroup(1, rates='stimadn(t)')
# I_pos 			= PoissonGroup(1, rates='stimpos(t)')
# G 				= NeuronGroup(3, model=eqs_neurons, threshold='v > 0.99', reset='v = 0', method = 'exact')
# G.tau			= 50*ms
# S_adn 			= Synapses(I_adn, G, 'w : 1', on_pre='v_post += w')
# S_adn.connect(i=0, j=np.arange(0,len(G)))
# S_adn.w 		= 0.01
# # S_adn.delay 	= np.array([0.0, 10.0])*ms
# S_adn.delay 	= np.array([0.0, 50.0, 100.0])*ms
# S_pos 			= Synapses(I_pos, G, 'w : 1', on_pre='v_post += w')
# S_pos.connect(i=0, j=np.arange(0,len(G)))
# S_pos.w 		= 0.6
# M2 = StateMonitor(G, 'v', record=True)
# out_mon 		= SpikeMonitor(G)
# adn_mon 		= SpikeMonitor(I_adn)
# pos_mon 		= SpikeMonitor(I_pos)
# duration 		= len(nang)*bin_size*ms
# run(duration, report = 'text')
# print(" Simulation finished")
# adn_spikes = adn_mon.spike_trains()[0]
# pos_spikes = pos_mon.spike_trains()[0]
# out_spikes = {n:out_mon.spike_trains()[n] for n in out_mon.spike_trains().keys()}

#####################################################################
# CROSS_CORRELATION OF SIMULATED NEURONS
#####################################################################

corr_simu = crossCorr(np.round(pos_spikes/ms), adn_spikes/ms, cc_bin_size, cc_bin_length/cc_bin_size)
corr_simu = pd.DataFrame(index = time_cc, data = corr_simu, columns = ['simu'])
corr_simu_delay = []
for k in out_spikes.keys():
	corr_simu_delay.append(crossCorr(np.round(out_spikes[k]/ms), np.round(adn_spikes/ms), cc_bin_size, cc_bin_length/cc_bin_size))
corr_simu_delay = np.array(corr_simu_delay)
corr_simu_delay = pd.DataFrame(columns = out_spikes.keys(), index = time_cc, data = corr_simu_delay.transpose())


# #############################################################################
# # TUNING CURVES BRIAN
# #############################################################################
data = pd.DataFrame()
f, bins_edge = np.histogram(np.round(adn_spikes/ms), int(duration / ms / bin_size), range = (0, duration/ms))
data['ADn'] = f 
f, bins_edge = np.histogram(np.round(pos_spikes/ms), int(duration / ms / bin_size), range = (0, duration/ms))
data['Pos'] = f 
for k in out_spikes.keys():
	f, bins_edge = np.histogram(np.round(out_spikes[k]/ms), int(duration / ms / bin_size), range = (0, duration/ms))
	data['Pos_'+str(k)] = f

tcurves_brian = pd.DataFrame()
x, tcurve, index = tuning_curve(nang, data['ADn'].values, 60, 1/float(bin_size*1e-3))
tcurves_brian['ADn'] = tcurve
x, tcurve, index = tuning_curve(nang, data['Pos'].values, 60, 1/float(bin_size*1e-3))
tcurves_brian['Pos'] = tcurve
tcurves_brian = tcurves_brian.set_index(x)
for k in out_spikes.keys():
	x, tcurve, index = tuning_curve(nang, data['Pos_'+str(k)].values, 60, 1/float(bin_size*1e-3))
	tcurves_brian['Pos_'+str(k)] = pd.DataFrame(data = tcurve, index = x)





# figure()
# plot(adn_mon.t, adn_mon.i, '.')
# plot(pos_mon.t, pos_mon.i+1, '.')
# plot(out_mon.t, out_mon.i+3, '.')


figure()
subplot(221)
plot(tcurves_brian['ADn'], label = 'brian Adn', linewidth = 3)
plot(adn, label = 'real Adn', linewidth = 3)
legend()
subplot(223)
# plot(dummy_tcurve, linewidth = 3, label = 'dummy Pos')
plot(tcurves_brian['Pos'], label = 'brian Pos', linewidth = 3)
plot(pos, label = 'real Pos', linewidth = 3)
# for k in out_spikes.keys():
# 	plot(tcurves_brian['Pos_'+str(k)], '--', linewidth = 3, label = str(k))
# legend(loc = 'best')

subplot(122)
plot(corr_simu, label = 'simu', linewidth = 3)
plot(corr_real, label = 'real', linewidth = 3)
for k in corr_simu_delay.columns:
	plot(corr_simu_delay[k], '--', linewidth = 3, label = str(k))
legend(loc = 'best')

show()

# for j in np.arange(0, 400, 100):
# 	figure()
# 	count = 1
# 	for i in np.arange(j, j+100):
# 		subplot(10,10,count)
# 		lb = "tau:"+str(float(G.tau[i])) + " w:" + str(S_pos.w[i])
# 		plot(corr_simu_delay[i], '--', linewidth = 3)
# 		plot(corr_real, label = 'real', linewidth = 3)
# 		plot(corr_simu, ':', label = 'simu', linewidth = 3)
# 		title(lb, fontsize = 5)
# 		# xticks([])
# 		# yticks([])
# 		count += 1
# show()


sys.exit()






##########################################################################
# TIME PEER PREDICTION
##########################################################################
import xgboost as xgb
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


adn_neuron = ['ADn']
pos_neuron = ['Pos']
out_neuron = ['Pos_0', 'Pos_1']

###########################################################################################
# create shifted spiking activity from -500 to 500 ms with index 0 to 40 (20 for t = 0 ms) for all ADn_neuron
# remove 20 points at the beginning and 20 points at the end
###########################################################################################
nb_bins = 1000/bin_size
duration = len(data)			
time_shifted = np.zeros(( duration-nb_bins, len(adn_neuron), nb_bins+1))
for n,i in zip(adn_neuron, range(len(adn_neuron))):	
	tmp = data[n]
	for j in range(0,nb_bins+1):
		# time_shifted[:,i,j] = tmp[40-j:duration-j]
		time_shifted[:,i,j] = tmp[j:duration-nb_bins+j]


combination = {}
for k in out_neuron:
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
num_round = 80

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
time_count = np.zeros((len(out_neuron), len(adn_neuron), nb_bins+1))
index = np.repeat(np.arange(len(adn_neuron)), nb_bins+1)
for n in thresholds.iterkeys():
	splits = thresholds[n]
	for s in splits.keys():
		time_count[int(n.split("_")[1]), index[int(s[1:])], int(s[1:])%(nb_bins+1)] = len(splits[s])

time_count = time_count.sum(1)

gain_value = np.zeros((len(out_neuron), len(adn_neuron), nb_bins+1))
index = np.repeat(np.arange(len(adn_neuron)), nb_bins+1)
for n in gain.iterkeys():
	g = gain[n]
	for s in g.keys():
		gain_value[int(n.split("_")[1]), index[int(s[1:])], int(s[1:])%(nb_bins+1)] = g[s]

# gain_value = gain_value.reshape(len(out_neuron)*len(adn_neuron), 41)
gain_value = gain_value.sum(1)
xgb_times = np.arange(-(cc_bin_length/2), (cc_bin_length/2)+bin_size, bin_size)
xgb_peaks = pd.DataFrame(index = xgb_times, data = (time_count*gain_value).transpose(), columns = np.arange(len(out_neuron)))


#####################################################################
# PLOT
#####################################################################

figure()
subplot(221)
plot(tcurves_brian['ADn'], label = 'brian Adn', linewidth = 3)
plot(adn, label = 'real Adn', linewidth = 3)
legend()
subplot(223)
plot(tcurves_brian['Pos'], label = 'brian Pos', linewidth = 3)
plot(pos, label = 'real Pos', linewidth = 3)
for k in out_spikes.keys():
	plot(tcurves_brian['Pos_'+str(k)], '--', linewidth = 3, label = str(k))
legend(loc = 'best')

subplot(222)
plot(corr_simu, label = 'simu', linewidth = 3)
plot(corr_real, label = 'real', linewidth = 3)
for k in corr_simu_delay.columns:
	plot(corr_simu_delay[k], '--', linewidth = 3, label = str(k))

subplot(224)
plot(xgb_peaks)


show()

