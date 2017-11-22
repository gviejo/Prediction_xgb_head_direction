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


bin_size = 25
time = np.arange(-500, 500+bin_size, bin_size)
matdata = scipy.io.loadmat("../data/sessions_nosmoothing_"+str(bin_size)+"ms/wake/boosted_tree.Mouse28-140313.mat")
n_adn 	= matdata['ADn'].shape[1]
n_pos 	= matdata['Pos'].shape[1]
ang 	= matdata['Ang'].flatten()
tcurves = []
columns = []
for k, n_neurons in zip(['ADn', 'Pos'],[n_adn, n_pos]):	
	for n in range(n_neurons):
		columns.append(k+"."+str(n))
		xbins, frate, index = tuning_curve(ang, matdata[k][:,n], 15, 1/(bin_size*1e-3))
		tcurves.append(frate)

tcurves = pd.DataFrame(index = xbins, columns = columns, data = np.array(tcurves).transpose())


to_keep = ['ADn.7', 'Pos.8']
adn = tcurves[to_keep[0]]
pos = tcurves[to_keep[1]]


############################################################################
# SIMULATION
############################################################################
from brian2 import *
set_device('cpp_standalone')
start_scope()

# eqs = '''
# dv/dt = -v/tau + sigma*xi*tau**-0.5 : volt
# I : 1
# tau : second
# '''
tau 			= 10*ms
sigma 			= .0
# eqs_neurons 	= '''
# dv/dt = -v / tau + sigma * (2 / tau)**.5 * xi : 1 (unless refractory)
# '''
eqs_neurons 	= 	'''
					dv/dt = -v / tau : 1 (unless refractory)
					'''

freq_steps_adn 	= adn.reindex(ang, method = 'nearest').values.flatten()
freq_steps_pos 	= pos.reindex(ang, method = 'nearest').values.flatten()

stimadn 		= TimedArray(freq_steps_adn * Hz, dt = float(bin_size) * ms)
stimpos 		= TimedArray(freq_steps_pos * Hz, dt = float(bin_size) * ms)

I_adn 			= PoissonGroup(1, rates='stimadn(t)')
I_pos 			= PoissonGroup(1, rates='stimpos(t)')
G 				= NeuronGroup(4, model=eqs_neurons, threshold='v > 1', reset='v = 0', refractory=0*ms, method='euler')
S_adn 			= Synapses(I_adn, G, 'w : 1', on_pre='v_post += w')
S_adn.connect(i=0, j=np.arange(0,4))
S_adn.w 		= '0.0'
S_adn.delay 	= 'j*50*ms'
S_pos 			= Synapses(I_pos, G, 'w : 1', on_pre='v_post += w')
S_pos.connect(i=0, j=np.arange(0,4))
S_pos.w 		= '1.2'
# S_pos.delay 	= '0*ms'
# M2 = StateMonitor(G, 'v', record=True)
out_mon 		= SpikeMonitor(G)
adn_mon 		= SpikeMonitor(I_adn)
pos_mon 		= SpikeMonitor(I_pos)
duration 		= len(ang)*bin_size*ms
run(duration, report = 'text')
# duration = 5*60.*second
# run(duration, profile = 'True')
print(" Simulation finished")

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

spikes = scipy.io.loadmat("../data/spikes.Mouse28-140313.mat")
spikes_ts = dict()
for k, n_neurons in zip(['ADn', 'Pos'],[n_adn, n_pos]):	
	tmp = spikes[k][0][0][0]
	for n in range(n_neurons):
		spikes_ts[k+"."+str(n)] = tmp[n][0][0,0][1][0][0][2].flatten()
		


# TO SHOW ALL CROSS CORR OF ALL PAIRS OF POS_ADN NEURONS
# corr_real = []
# real_adn_neuron = ["ADn."+str(i) for i in np.arange(n_adn)]
# real_pos_neuron = ["Pos."+str(j) for j in np.arange(n_pos)]
# count = 1
# for m in real_adn_neuron:
# 	for n in real_pos_neuron:		
# 		C = crossCorr(spikes_ts[n]*1000, spikes_ts[m]*1000, bin_size, 1000/bin_size)
# 		# frate = float(len(spikes_ts[m]))/float(spikes_ts[m][-1] - spikes_ts[m][0])
		
# 		subplot(n_adn, n_pos, count)
# 		plot(C)
# 		title(m+" "+n, fontsize = 5)
# 		xticks([])
# 		yticks([])
# 		axvline(20)
# 		count += 1
# 		corr_real.append(C)
# show()

corr_real = crossCorr(spikes_ts[to_keep[1]]*1000, spikes_ts[to_keep[0]]*1000, bin_size, 1000/bin_size)
corr_real = pd.DataFrame(index = time, data = corr_real, columns = ['real'])

#####################################################################
# CROSS_CORRELATION OF SIMULATED NEURONS
#####################################################################
def crossCorr2(t1, t2, binsize, nbins):
	'''
		Slow crossCorr
	'''
	window = np.arange(-binsize*(nbins/2),binsize*(nbins/2)+2*binsize,binsize) - (binsize/2.)
	allcount = np.zeros(nbins+1)
	for e in t1:
		mwind = window + e
		index = np.digitize(t2, mwind)
		# index larger than 2 and lower than mwind.shape[0]-1
		# count each occurences 
		count = np.array([np.sum(index == i) for i in range(1,mwind.shape[0])])
		allcount += np.array(count)
	allcount = allcount/(float(len(t1))*binsize / 1000)
	return allcount

adn_spikes = np.array(adn_mon.spike_trains()[0])
pos_spikes = np.array(pos_mon.spike_trains()[0])
adn_spikes += spikes_ts[to_keep[0]][0]
pos_spikes += spikes_ts[to_keep[1]][0]

out_spikes = out_mon.spike_trains()
corr_simu = crossCorr2(np.array(pos_spikes)*1000., np.array(adn_spikes)*1000., bin_size, 1000/bin_size)
corr_simu = pd.DataFrame(index = time, data = corr_simu, columns = ['simu'])



corr_simu_delay = []
for k in out_spikes.keys():
	corr_simu_delay.append(crossCorr(np.array(out_spikes[k])*1000., np.array(adn_spikes)*1000., bin_size, 1000/bin_size))
corr_simu_delay = np.array(corr_simu_delay)
corr_simu_delay = pd.DataFrame(columns = S_adn.delay, index = time, data = corr_simu_delay.transpose())



# tuning_curves
data = pd.DataFrame()
f, bins_edge = np.histogram(adn_spikes[0]/ms, int(duration / ms / bin_size), range = (0, duration/ms))
data['ADn'] = f 
f, bins_edge = np.histogram(pos_spikes[0]/ms, int(duration / ms / bin_size), range = (0, duration/ms))
data['Pos'] = f 
for k in out_spikes.keys():
	f, bins_edge = np.histogram(out_spikes[k]/ms, int(duration / ms / bin_size), range = (0, duration/ms))
	data['Pos_'+str(S_adn.delay[k])] = f

tcurves_brian = pd.DataFrame()
x, tcurve, index = tuning_curve(ang, data['ADn'].values, 15, 1/float(bin_size*1e-3))
tcurves_brian['ADn'] = tcurve
x, tcurve, index = tuning_curve(ang, data['Pos'].values, 15, 1/float(bin_size*1e-3))
tcurves_brian['Pos'] = tcurve
tcurves_brian = tcurves_brian.set_index(x)
for k in out_spikes.keys():
	x, tcurve, index = tuning_curve(ang, data['Pos_'+str(S_adn.delay[k])].values, 15, 1/float(bin_size*1e-3))
	tcurves_brian['Pos_'+str(S_adn.delay[k])] = pd.DataFrame(data = tcurve, index = x)


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
plot(tcurves_brian['Pos'], label = 'brian Pos', linewidth = 3)
plot(pos, label = 'real Pos', linewidth = 3)
for k in S_adn.delay:
	plot(tcurves_brian['Pos_'+str(k)], '--', linewidth = 3, label = str(k))

legend()
subplot(122)
plot(corr_simu, label = 'simu', linewidth = 3)
plot(corr_real, label = 'real', linewidth = 3)
for k in corr_simu_delay.columns:
	plot(corr_simu_delay[k], '--', linewidth = 3, label = str(k)+" ms")
legend()

show()

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

bin_size = 5 # ms
data = pd.DataFrame()
out_spikes = out_mon.spike_trains()

for k in adn_spikes.keys():
	f, bins_edge = np.histogram(adn_spikes[k]/ms, int(duration / ms / bin_size), range = (0, duration/ms))
	data['ADn'+'.'+str(k)] = f / (bin_size*1e-3)

for k in pos_spikes.keys():
	f, bins_edge = np.histogram(pos_spikes[k]/ms, int(duration / ms / bin_size), range = (0, duration/ms))
	data['Pos'+'.'+str(k)] = f / (bin_size*1e-3)

for k in out_spikes.keys():
	f, bins_edge = np.histogram(out_spikes[k]/ms, int(duration / ms / bin_size), range = (0, duration/ms))
	data['Out'+'.'+str(k)] = f / (bin_size*1e-3)


adn_neuron = [n for n in data.keys() if 'ADn' in n]
pos_neuron = [n for n in data.keys() if 'Pos' in n]
out_neuron = [n for n in data.keys() if 'Out' in n]

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
time_count = np.zeros((len(out_neuron), len(adn_neuron), nb_bins+1))
index = np.repeat(np.arange(len(adn_neuron)), nb_bins+1)
for n in thresholds.iterkeys():
	splits = thresholds[n]
	for s in splits.keys():
		time_count[int(n.split(".")[1]), index[int(s[1:])], int(s[1:])%(nb_bins+1)] = len(splits[s])

time_count = time_count.sum(1)

gain_value = np.zeros((len(out_neuron), len(adn_neuron), nb_bins+1))
index = np.repeat(np.arange(len(adn_neuron)), nb_bins+1)
for n in gain.iterkeys():
	g = gain[n]
	for s in g.keys():
		gain_value[int(n.split(".")[1]), index[int(s[1:])], int(s[1:])%(nb_bins+1)] = g[s]

# gain_value = gain_value.reshape(len(out_neuron)*len(adn_neuron), 41)
gain_value = gain_value.sum(1)




time = np.arange(-500, 500+bin_size, bin_size)
xgb_peaks = pd.DataFrame(index = time, data = (time_count*gain_value).transpose(), columns = S_adn.delay)


corr_out = []
for k in out_spikes.keys():
	corr_out.append(crossCorr(np.array(out_spikes[k]/ms), np.array(adn_spikes[0]/ms), bin_size, 1000/bin_size))
corr_out = np.array(corr_out)
corr_out = pd.DataFrame(columns = S_adn.delay, index = time, data = corr_out.transpose())
corr_out[corr_out.isnull()] = 0.0


#####################################################################
# PLOT
#####################################################################

figure()
ax = subplot(211)
plot(corr_real, label = 'real')
plot(corr_simu, label = 'simu/no delay')
title("Cross correlation")

ax = subplot(212)
plot(time, xgb_peaks, label = corr_simu.columns.values)
legend()
title("XGB")
show()