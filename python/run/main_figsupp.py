#!/usr/bin/env python

'''
    File name: main_fig_supp.py
    Author: Guillaume Viejo
    Date created: 12/05/2017    
    Python Version: 2.7

Figure supp

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

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################



final_data = {}
for ses in os.listdir("../data/sessions/wake/"):
#####################################################################
# DATA LOADING
#####################################################################
    wake_data = scipy.io.loadmat(os.path.expanduser('../data/sessions_nosmoothing_200ms/wake/'+ses))

#####################################################################
# DATA ENGINEERING
#####################################################################
    data            =   pd.DataFrame()
    data['time']    =   np.arange(len(wake_data['Ang']))      # TODO : import real time from matlab script
    data['ang']     =   wake_data['Ang'].flatten()            # angular direction of the animal head
    data['x']       =   wake_data['X'].flatten()              # x position of the animal 
    data['y']       =   wake_data['Y'].flatten()              # y position of the animal
    data['vel']     =   wake_data['speed'].flatten()          # velocity of the animal 
    # Firing data
    for i in xrange(wake_data['Pos'].shape[1]): data['Pos'+'.'+str(i)] = wake_data['Pos'][:,i]
    for i in xrange(wake_data['ADn'].shape[1]): data['ADn'+'.'+str(i)] = wake_data['ADn'][:,i]
    # data['rand0']    =   np.random.uniform(0, 2*np.pi, len(wake_data['Ang']))
    # data['rand1']    =   np.random.uniform(0, 2*np.pi, len(wake_data['Ang']))
    # data['rand2']    =   np.random.uniform(0, 2*np.pi, len(wake_data['Ang']))

    data['rand0']    =   np.ones(len(wake_data['Ang']))
    data['rand1']    =   np.ones(len(wake_data['Ang']))
    data['rand2']    =   np.ones(len(wake_data['Ang']))

########################################################################
# COMBINATIONS DEFINITIONS
########################################################################
    combination = {
        'all':  {
                'features'  :   ['rand0', 'rand1', 'ang', 'rand2'],
                'targets'   :   [i for i in list(data) if i.split(".")[0] == 'ADn']+[i for i in list(data) if i.split(".")[0] == 'Pos'], 
                }

    }

#####################################################################
# LEARNING XGB
#####################################################################
    params = {'objective': "count:poisson", #for poisson output
        'eval_metric': "poisson-nloglik", #loglikelihood loss
        'seed': 2925, #for reproducibility
        'silent': 1,
        'learning_rate': 0.1,
        'min_child_weight': 2, 'n_estimators': 1,
        'subsample': 0.6, 'max_depth': 5, 'gamma': 0.5}

    def func(x, a, b, c, d):
        return a*np.exp(-(x/b)+c)+d

    num_round = 1000
    tmp = []
    X = data[combination['all']['features']].values
    for n in combination['all']['targets']:
        Y = data[n]
        dtrain = xgb.DMatrix(np.vstack(X), label = np.vstack(Y))
        for i in xrange(1, 300):            
            print i
            bst = xgb.train(params, dtrain, i)
            tmp.append(bst.get_score(importance_type = 'gain')['f0'])
            # w = bst.get_score(importance_type = 'weight')
            tmp.append([w[f] for f in ['f0', 'f1', 'f2', 'f3']])
        tmp = np.array(tmp)
        tmp = tmp/np.vstack(np.sum(tmp, 1)).astype('float')
        [plot(tmp[:,i], label = i) for i in xrange(4)]
        legend()

        sys.exit()

        x = np.arange(1, 500)
        a = scipy.optimize.curve_fit(func, x, tmp)
        sys.exit()
        
