import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
# from functions import *
# from pylab import *
import ipyparallel
import os, sys

data_directory = "../data/spike_timing/wake"
datasets = os.listdir(data_directory)

session = 'spike_timing.Mouse20-130520.mat'

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

	for j in range(nbins):
		C[j] = C[j] / (nt1 * binsize)

	return C

data = {}

data[session] = {}

# print(session, len(data.keys())/float(len(datasets)))


###############################################################################################################
# SPIKE
###############################################################################################################
spikedata = scipy.io.loadmat(data_directory+'/'+session)

n_thalamus = len(spikedata['adn'][0][0][0])
n_postsub = len(spikedata['pos'][0][0][0])
spikes_thalamus = [spikedata['adn'][0][0][0][i][0][0][0][1][0][0][2] for i in range(n_thalamus)]
spikes_postsub = {i:spikedata['pos'][0][0][0][i][0][0][0][1][0][0][2] for i in range(n_postsub)}



# ##############################################################################################################
# # CROSS CORRELATION
# ##############################################################################################################
# pool spike timing of thalamus
spikes_thalamus = np.sort(np.vstack(spikes_thalamus).transpose()[0])
order = ['wake', 'rem', 'sleep']


# for i in range(len(order)):
for i in [0]:
	data[session][order[i]] = {}
	
	
	# cross correlation for each neuron in post-sub
	for n in spikes_postsub.keys():
		bin_size = 0.005
		nb_bins = 200
		C = crossCorr(spikes_postsub[n], spikes_thalamus, bin_size, nb_bins)
		data[session][order[i]][n] = C

		sys.exit()
	
