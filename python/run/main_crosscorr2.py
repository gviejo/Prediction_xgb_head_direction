# USING data/spike_timing 

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
# from functions import *
# from pylab import *
import ipyparallel
import os, sys

data_directory = "/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData"
datasets = np.loadtxt(data_directory+'/datasets_PosData.list', delimiter = '\n', dtype = str, comments = '#')

clients = ipyparallel.Client()
print(clients.ids)
dview = clients.direct_view()

def compute_cross_correlation(episode):

	import numpy as np
	import pandas as pd
	# from matplotlib.pyplot import plot,show,draw
	import scipy.io
	import os, sys

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


	data_directory = "../data/spike_timing/"+episode
	datasets = os.listdir(data_directory)

	data = {}

	for session in datasets:

		s = session.split(".")[1]
		data[s] = []

		###############################################################################################################
		# SPIKE
		###############################################################################################################
		spikedata = scipy.io.loadmat(data_directory+'/'+session)

		n_thalamus = len(spikedata['adn'][0][0][0])
		n_postsub = len(spikedata['pos'][0][0][0])
		spikes_thalamus = [spikedata['adn'][0][0][0][i][0][0][0][1][0][0][2] for i in range(n_thalamus)]
		spikes_postsub = {i:spikedata['pos'][0][0][0][i][0][0][0][1][0][0][2] for i in range(n_postsub)}


		##############################################################################################################
		# CROSS CORRELATION
		##############################################################################################################
		# pool spike timing of thalamus
		spikes_thalamus = np.sort(np.vstack(spikes_thalamus).transpose()[0])

			
		# cross correlation for each neuron in post-sub
		for n in spikes_postsub.keys():
			bin_size = 0.005
			nb_bins = 200
			C = crossCorr(spikes_postsub[n], spikes_thalamus, bin_size, nb_bins)
			# mean_firing_rate = float(len(ts_thalamus))/float(np.sum(np.sum(ep[:,1] - ep[:,0])))
			data[s].append(C)

		data[s] = np.array(data[s])

	
	return data



# a = dview.map_sync(compute_cross_correlation, datasets)

a = compute_cross_correlation('wake')







##############################################################################################################
# PLOT
##############################################################################################################
from pylab import *

times = np.arange(-500, 505, 5)

figure()
tmp = []
for s in a.keys():
		
	plot(times, a[s].transpose(), '-', color = 'grey', alpha = 0.5)

	if len(a[s]):
		tmp.append(a[s])

tmp = np.vstack(tmp)

plot(times, tmp.mean(0))

fill_between(times, tmp.mean(0)-tmp.var(0), tmp.mean(0)+tmp.var(0), color = 'blue', alpha = 1)

show()