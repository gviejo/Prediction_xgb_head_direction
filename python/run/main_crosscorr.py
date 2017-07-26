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

def compute_cross_correlation(session):

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


	data_directory = "/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/HDCellData"

	data = {}


	data[session] = {}

	# print(session, len(data.keys())/float(len(datasets)))

	###############################################################################################################
	# GENERAL INFO
	###############################################################################################################
	generalinfo = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/GeneralInfo.mat')

	###############################################################################################################
	# SHANK INFO
	###############################################################################################################
	shankStructure = {}
	for k,i in zip(generalinfo['shankStructure'][0][0][0][0],range(len(generalinfo['shankStructure'][0][0][0][0]))):
		if len(generalinfo['shankStructure'][0][0][1][0][i]):
			shankStructure[k[0]] = generalinfo['shankStructure'][0][0][1][0][i][0]
		else :
			shankStructure[k[0]] = []

	###############################################################################################################
	# SPIKE
	###############################################################################################################
	spikedata = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/SpikeData.mat')
	shank = spikedata['shank']
	shankIndex_thalamus = np.where(shank == shankStructure['thalamus'])[0]
	shankIndex_postsub 	= np.where(shank == shankStructure['postsub'])[0]

	nb_channels = len(spikedata['S'][0][0][0])
	spikes_thalamus = []	
	for i in shankIndex_thalamus:	
		spikes_thalamus.append(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2])
	spikes_postsub = []
	for i in shankIndex_postsub:	
		spikes_postsub.append(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2])		

	###############################################################################################################
	# BEHAVIORAL EPOCHS
	###############################################################################################################
	behepochs = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/BehavEpochs.mat')
	sleep_pre_ep = behepochs['sleepPreEp'][0][0]
	sleep_pre_ep = np.hstack([sleep_pre_ep[1],sleep_pre_ep[2]])
	sleep_pre_ep_index = behepochs['sleepPreEpIx'][0]
	sleep_post_ep = behepochs['sleepPostEp'][0][0]
	sleep_post_ep = np.hstack([sleep_post_ep[1],sleep_post_ep[2]])
	sleep_post_ep_index = behepochs['sleepPostEpIx'][0]
	if len(sleep_pre_ep) and len(sleep_post_ep):
		sleep_ep = np.vstack((sleep_pre_ep, sleep_post_ep))
	elif len(sleep_pre_ep):
		sleep_ep = sleep_pre_ep
	elif len(sleep_post_ep):
		sleep_ep = sleep_post_ep
	# merge sleep ep
	corresp = sleep_ep[1:,0].astype('int') == sleep_ep[0:-1,1].astype('int')
	start = sleep_ep[0,0]
	tmp = []
	for i,j in zip(corresp,range(len(corresp))):
		if not i:
			stop = sleep_ep[j,1]
			tmp.append([start, stop])
			start = sleep_ep[j+1,0]
	tmp.append([start, sleep_ep[-1,1]])
	sleep_ep = np.array(tmp)
	# wake ep
	wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
	# restrict linear speed tsd to wake ep
	# linear_speed_tsd = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/linspeed.mat')['speed']
	# tmp = []
	# for e in wake_ep:
	# 	start, stop = e
	# 	for t in linear_speed_tsd:
	# 		if t[0] > start and t[0] < stop:
	# 			tmp.append(t)
	# linear_speed_tsd = np.array(tmp)
	# index = (linear_speed_tsd[:,1] > 2.0)*1.0
	# start = np.where((index[1:]-index[0:-1] == 1))[0]+1
	# stop = np.where((index[1:]-index[0:-1] == -1))[0]
	# if len(start) == len(stop):
	# 	linear_speed_ep = np.vstack([linear_speed_tsd[start,0],linear_speed_tsd[stop,0]]).transpose()
	# else:
	# 	m = np.min([len(start), len(stop)])
	# 	linear_speed_ep = np.vstack([linear_speed_tsd[start[0:m],0],linear_speed_tsd[stop[0:m],0]]).transpose()
	# # restrict wake ep to speed > 2cm / s
	# wake_ep = linear_speed_ep

	# load SWS EP 
	if session.split("/")[1]+'.sts.SWS' in os.listdir(data_directory+'/'+session+'/'):
		sws = np.genfromtxt(data_directory+'/'+session+'/'+session.split("/")[1]+'.sts.SWS')		
		sws_ep = sws/float(sampling_freq)
	elif session.split("/")[1]+'-states.mat' in os.listdir(data_directory+'/'+session+'/'):
		sws = scipy.io.loadmat(data_directory+'/'+session+'/'+session.split("/")[1]+'-states.mat')['states'][0]
		index = np.logical_or(sws == 2, sws == 3)*1.0
		index = index[1:] - index[0:-1]
		start = np.where(index == 1)[0]+1
		stop = np.where(index == -1)[0]
		if len(start) == len(stop):
			sws_ep = np.hstack((np.vstack(start),
							np.vstack(stop))).astype('float')
		else :
			m = np.min([len(start), len(stop)])
			sws_ep = np.hstack((np.vstack(start[0:m]),
							np.vstack(stop[0:m]))).astype('float')

	# load REM EP
	if session.split("/")[1]+'.sts.REM' in os.listdir(data_directory+'/'+session+'/'):
		rem = np.genfromtxt(data_directory+'/'+session+'/'+session.split("/")[1]+'.sts.REM')
		rem_ep = rem/float(sampling_freq)
	elif session.split("/")[1]+'-states.mat' in os.listdir(data_directory+'/'+session+'/'):
		rem = scipy.io.loadmat(data_directory+'/'+session+'/'+session.split("/")[1]+'-states.mat')['states'][0]
		index = (rem == 5)*1.0
		index = index[1:] - index[0:-1]
		rem_ep = np.hstack((np.vstack(np.where(index == 1)[0]),
							np.vstack(np.where(index == -1)[0]))).astype('float')

	# restrict sws_ep and rem_ep by sleep_ep
	tmp1 = []
	tmp2 = []
	for e in sleep_ep:
		start, stop = e
		for s in sws_ep:
			substart, substop = s
			if substart > start and substop < stop:
				tmp1.append(s)
		for s in rem_ep:
			substart, substop = s
			if substart > start and substop < stop:
				tmp2.append(s)		
	sws_ep = np.array(tmp1)
	rem_ep = np.array(tmp2)



	##############################################################################################################
	# CROSS CORRELATION
	##############################################################################################################
	# pool spike timing of thalamus
	spikes_thalamus = np.sort(np.vstack(spikes_thalamus).transpose()[0])
	order = ['wake', 'rem', 'sleep']
	episode = [wake_ep, rem_ep, sws_ep]

	for i in range(len(order)):
		data[session][order[i]] = {}
		ep = episode[i]
		# restrict spikes to ep for both thalamus and postsub
		ts_thalamus = []
		ts_postsub	= {n:[] for n in range(len(spikes_postsub))}
		for e in ep:
			start, stop = e
			for t in spikes_thalamus:
				if t > start and t < stop:
					ts_thalamus.append(t)
			for n in range(len(spikes_postsub)):
				for t in spikes_postsub[n]:
					if t > start and t < stop:
						ts_postsub[n].append(t)
		ts_thalamus = np.array(ts_thalamus)
		for n in ts_postsub.keys(): 
			ts_postsub[n] = np.array(ts_postsub[n]).flatten()


		# cross correlation for each neuron in post-sub
		for n in ts_postsub.keys():
			bin_size = 0.005
			nb_bins = 200
			C = crossCorr(ts_postsub[n], ts_thalamus, bin_size, nb_bins)
			data[session][order[i]][n] = C

		
		return data


a = dview.map_sync(compute_cross_correlation, datasets)