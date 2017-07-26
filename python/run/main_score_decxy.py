#!/usr/bin/env python

'''
    File name: main_fig5.py
    Author: Guillaume Viejo
    Date created: 12/05/2017    
    Python Version: 2.7

Plot figure 5 for decodage

'''

import warnings
import pandas as pd
import scipy.io
import numpy as np
# Should not import fonctions if already using tensorflow for something else
import sys, os
import itertools
import cPickle as pickle
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.linear_model import LinearRegression

os.system("scp -r viejo@guillimin.hpc.mcgill.ca:~/results_decoding_xyang/ ../data/")

def gaussFilt(X, wdim = (1,)):
    '''
        Gaussian Filtering in 1 or 2d.      
        Made to fit matlab
    '''
    from scipy.signal import gaussian

    if len(wdim) == 1:
        from scipy.ndimage.filters import convolve1d
        l1 = len(X)
        N1 = wdim[0]*10
        S1 = (N1-1)/float(2*5)
        gw = gaussian(N1, S1)
        gw = gw/gw.sum()
        #convolution
        filtered_X = convolve1d(X, gw)
        return filtered_X   
    elif len(wdim) == 2:
        from scipy.signal import convolve2d
        def conv2(x, y, mode='same'):
            return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)           

        l1, l2 = X.shape
        N1, N2 = wdim       
        # create bordered matrix
        Xf = np.flipud(X)
        bordered_X = np.vstack([
                np.hstack([
                    np.fliplr(Xf),Xf,np.fliplr(Xf)
                ]),
                np.hstack([
                    np.fliplr(X),X,np.fliplr(X)
                ]),
                np.hstack([
                    np.fliplr(Xf),Xf,np.fliplr(Xf)
                ]),
            ])
        # gaussian windows
        N1 = N1*10
        N2 = N2*10
        S1 = (N1-1)/float(2*5)
        S2 = (N2-1)/float(2*5)
        gw = np.vstack(gaussian(N1,S1))*gaussian(N2,S2)
        gw = gw/gw.sum()
        # convolution
        filtered_X = conv2(bordered_X, gw, mode ='same')
        return filtered_X[l1:l1+l1,l2:l2+l2]
    else :
        print("Error, dimensions larger than 2")
        return


data = {}

for f in os.listdir("../data/results_decoding_xyang/"):
    data[f.split(".")[1]] = pickle.load(open("../data/results_decoding_xyang/"+f, 'rb'))

from pylab import *

for ses in data.keys():
    figure()
    ax1 = subplot(311)
    plot(data[ses]['real'][:,0], color = 'black')
    plot(data[ses]['allADn']['xgb_decodage'][:,0])
    ylabel('angle')

    ax2 = subplot(312,sharex=ax1)
    plot(data[ses]['real'][:,1], color = 'black')
    # plot(data[ses]['allADn']['xgb_decodage'][:,1], color = 'red')
    plot(gaussFilt(data[ses]['allPos']['xgb_decodage'][:,1], (6,)), color = 'blue')
    ylabel('x')

    ax3 = subplot(313,sharex=ax1)
    plot(data[ses]['real'][:,2], color = 'black')
    # plot(data[ses]['allADn']['xgb_decodage'][:,2], color = 'red')
    plot(gaussFilt(data[ses]['allPos']['xgb_decodage'][:,2], (6,)), color = 'blue')
    ylabel('y')

    # figure()
    # plot(data[ses]['real'][:,1], data[ses]['real'][:,2], color = 'black')
    show()





# score = {m:{k:[] for k in ['samADn', 'samPos', 'allADn', 'shuPos', 'allPos', 'shuADn']} for m in ['bay_decodage', 'xgb_decodage']}
score = {m:{k:[] for k in ['allADn', 'allPos']} for m in ['bay_decodage', 'xgb_decodage']}

tmp = 100000.0
best = ()
for ses in data.iterkeys():
    real = data[ses]['real']
    for m in score.iterkeys():
        for k in score[m].iterkeys():
            if len(data[ses][k].keys()) == 2:
                predi = data[ses][k][m]
                score[m][k].append(np.sum(np.sqrt(np.power(predi - real, 2).sum(1))))
                if np.sum(np.sqrt(np.power(predi - real, 2).sum(1))) < tmp:
                    tmp = np.sum(np.sqrt(np.power(predi - real, 2).sum(1)))
                    best =  (ses, m, k, np.sum(np.sqrt(np.power(predi - real, 2).sum(1))))

meanscore = {}
for m in score.iterkeys():
    meanscore[m] = {}
    for g in ['ADn', 'Pos']:
        meanscore[m][g] = {}
        for k in score[m].iterkeys():
            if g in k:
                meanscore[m][g][k[0:3]] = [np.mean(score[m][k]), np.var(score[m][k])]

from pylab import *
figure()

colors_ = {'ADn':'#EE6C4D', 'Pos':'#3D5A80'}
labels_ = {'ADn':'Antero-dorsal nucleus', 'Pos':'Post-subiculum'}
x = [0.0]
y = []
e = []
group = []

for m in ['bay_decodage', 'xgb_decodage']:
    for g in ['ADn', 'Pos']:
        for k in ['sam', 'all', 'shu']:
            y.append(meanscore[m][g][k][0])
            e.append(meanscore[m][g][k][1])
            x.append(x[-1]+1)
            group.append(g)

        x[-1] += 1
    x[-1] += 1

x = np.array(x)[0:-1]
y = np.array(y)
e = np.array(e)
group = np.array(group)


for m in ['ADn', 'Pos']:
    bar(x[group == m], y[group == m], color = colors_[m], label = labels_[m])

legend()
xticks([x[2], x[8]], ['Bayesian', 'XGB'])


show()


