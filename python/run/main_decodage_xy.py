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

#######################################################################
# FONCTIONS DEFINITIONS
#######################################################################
def tuning_curve(x, f, nb_bins):    
    bins = np.linspace(x.min(), x.max()+1e-8, nb_bins+1)
    index = np.digitize(x, bins).flatten()    
    tcurve = np.array([np.sum(f[index == i]) for i in xrange(1, nb_bins+1)])    
    occupancy = np.array([np.sum(index == i) for i in xrange(1, nb_bins+1)])
    tcurve = (tcurve/occupancy)*5.0
    x = bins[0:-1] + (bins[1]-bins[0])/2.    
    return (x, tcurve)

def bay_decodage(Xr, Yr, Xt, nb_bins = 10):
    # Xr firing rate
    # Yr position in 2d  
    # predicting sequentially for each dimension
    tau = 0.200

    Yt = np.zeros((Xt.shape[0], Yr.shape[1]))
    for j in xrange(2):
        Yrr = Yr[:,j]
        pattern = np.zeros((nb_bins,Xr.shape[1]))
        for k in xrange(Xr.shape[1]):            
            theta, tuning = tuning_curve(Yrr.flatten(), Xr[:,k], nb_bins)
            pattern[:,k] = tuning

        Yhat = np.zeros((Xt.shape[0], nb_bins))        
        tmp = np.exp(-tau*pattern.sum(1))    
        for i in xrange(Yhat.shape[0]):
            Yhat[i] = tmp * np.prod(pattern**(np.tile(Xt[i], (nb_bins, 1))), 1)

        index = np.argmax(Yhat, 1)
        Yt[:,j] = theta[index]
    return Yt

def xgb_decodage(Xr, Yr, Xt):      
    # order is [x, y]    
    nbins_xy = 10
    index = np.arange(nbins_xy*nbins_xy).reshape(nbins_xy,nbins_xy)
    # binning pos
    posbins = np.linspace(0, 1+1e-8, nbins_xy+1)
    xposindex = np.digitize(Yr[:,0], posbins).flatten()-1
    yposindex = np.digitize(Yr[:,1], posbins).flatten()-1
    # setting class from index
    clas = np.zeros(Yr.shape[0])
    for i in xrange(Yr.shape[0]):
        clas[i] = index[xposindex[i],yposindex[i]]
    
    dtrain = xgb.DMatrix(Xr, label=clas)
    dtest = xgb.DMatrix(Xt)

    params = {'objective': "multi:softprob",
    'eval_metric': "mlogloss", #loglikelihood loss
    'seed': 2925, #for reproducibility
    'silent': 1,
    'learning_rate': 0.05,
    'min_child_weight': 2, 
    'n_estimators': 1000,
    # 'subsample': 0.5,
    'max_depth': 5, 
    'gamma': 0.5,
    'num_class':index.max()+1}

    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    
    ymat = bst.predict(dtest)

    pclas = np.argmax(ymat, 1)
    x, y = np.mgrid[0:nbins_xy,0:nbins_xy]
    clas_to_index = np.vstack((x.flatten(), y.flatten())).transpose()
    Yp = clas_to_index[pclas]
    # returning real position
    real = np.zeros(Yp.shape)    
    xy = posbins[0:-1] + (posbins[1]-posbins[0])/2.        
    real[:,0] = xy[Yp[:,0]]
    real[:,1] = xy[Yp[:,1]]
    return real
    # return x[np.argmax(ymat,1)]

def fit_cv(X, Y, algorithm, n_cv=10, verbose=1):
    if np.ndim(X)==1:
        X = np.transpose(np.atleast_2d(X))
    cv_kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
    skf  = cv_kf.split(X)    
    Y_hat=np.zeros((len(Y),Y.shape[1]))
    
    for idx_r, idx_t in skf:        
        Xr = X[idx_r, :]
        Yr = Y[idx_r]
        Xt = X[idx_t, :]
        Yt = Y[idx_t]           
        Yt_hat = eval(algorithm)(Xr, Yr, Xt)         
        Y_hat[idx_t] = Yt_hat
        
    return Y_hat


def test_decodage(features, targets, learners):    
    X = data[features].values
    Y = data[targets].values
    Models = {method:{'PR2':[],'Yt_hat':[]} for method in learners}
    learners_ = list(learners)
    print learners_

        
    for method in learners_:    
        print('Running '+method+'...')                              
        Models[method] = fit_cv(X, Y, method, n_cv = 4)


    return Models

def find_featuresconditions(treelist, nclass, nround):
    treelist = np.array(treelist)
    treelist = treelist.reshape(nround, nclass)
    clas_to_tree = {}
    for i in xrange(nclass):
        clas_to_tree[i] = {}
        for j in xrange(nround):
            clas_to_tree[i][j] = {}
            tree = treelist[j,i]
            tmp = tree.split("\t")
            # searching for each leaves
            leaves = {}
            for l in tmp:
                if len(l):
                    if 'leaf' in l:
                        leaves[l.split("\n")[0]] = []
                        tmp.remove(l)

            for l in leaves.keys():   
                leafnum = l[0]
                leaves[l].append(leafnum)
                last = leaves[l][-1]
                while int(last) > 0:
                    for t in tmp:
                        if '='+last in t:
                            leaves[l].append(t[0])       
                    last = leaves[l][-1]

            # ordering conditions
            conditions = {}
            for l in leaves.iterkeys():
                value = float(l.split("=")[1])
                conditions[value] = []
                nodes = leaves[l]
                for n in nodes[0:-1]:
                    for t in tmp:
                        if 'yes='+n in t:
                            cond = t.split("[")[1].split("]")[0]
                            conditions[value].append(cond)
                        elif 'no='+n in t:
                            cond = t.split("[")[1].split("]")[0].replace("<", ">")
                            conditions[value].append(cond)
            clas_to_tree[i][j] = conditions
    return clas_to_tree

def find_conditions(clas_to_tree, X):
    conditions = {}
    for c in clas_to_tree.iterkeys(): # Class
        conditions[c] = {}
        for r in clas_to_tree[c].iterkeys(): # Round
            maxleaf = np.max(clas_to_tree[c][r].keys())
            thresholds = clas_to_tree[c][r][maxleaf]
            for t in thresholds:
                feature = t.replace(">",":").replace("<",":").split(":")[0] 
                value = float(t.replace(">",":").replace("<",":").split(":")[1])
                if not conditions[c].has_key(feature):                                                                
                    max_firing_rate = X[:,int(feature[1:])].max() 
                    firing_rate_steps = np.arange(0, max_firing_rate+1, 0.5)
                    conditions[c][feature] = np.vstack((firing_rate_steps, np.zeros(len(firing_rate_steps)))).transpose()
                if "<" in t:
                    conditions[c][feature][conditions[c][feature][:,0] > value,1] += 1.0
                elif ">" in t:
                    conditions[c][feature][conditions[c][feature][:,0] < value,1] += 1.0
    return conditions

final_data = {}
for ses in os.listdir("../data/sessions_nosmoothing_200ms_allneuron/wake/"):
# for ses in ['boosted_tree.Mouse25-140131.mat']:
#####################################################################
# DATA LOADING
#####################################################################
    wake_data = scipy.io.loadmat(os.path.expanduser('../data/sessions_nosmoothing_200ms_allneuron/wake/'+ses))
    adn = wake_data['ADn'].shape[1]
    pos = wake_data['Pos'].shape[1]
        
    if adn >= 10 and pos >= 10:   
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
        # Let's normalize x and y
        for i in ['x', 'y']:
            data[i] = data[i]-np.min(data[i])
            data[i] = data[i]/np.max(data[i])
        # SHUFFLED DATA
        for g in ['Pos', 'ADn']:        
            tmp = np.copy(wake_data[g])
            for i in xrange(tmp.shape[1]):
                np.random.shuffle(tmp[:,i])
                data[g+'.'+str(i)+'.s'] = tmp[:,i]

########################################################################
# COMBINATIONS DEFINITIONS
#######################################################################
        # take the same number of feature when comparing Pos and ADn
        nmin = np.min([len([i for i in list(data) if i.split(".")[0] == 'Pos' and not '.s' in i]), len([i for i in list(data) if i.split(".")[0] == 'ADn' and not '.s' in i])])
        combination = {
            'allPos':  {
                    'targets'  :   ['x', 'y'],
                    'features'   :   [i for i in list(data) if i.split(".")[0] == 'Pos' and not '.s' in i], 
                    },          
            'allADn':  {
                    'targets'  :   ['x', 'y'],
                    'features'   :   [i for i in list(data) if i.split(".")[0] == 'ADn' and not '.s' in i],
                    },
            'samPos':  {
                    'targets'  :   ['x', 'y'],
                    'features'   :   np.random.choice([i for i in list(data) if i.split(".")[0] == 'Pos' and not '.s' in i], nmin, replace = False)
                    },          
            'samADn':  {
                    'targets'  :   ['x', 'y'],
                    'features'   :   np.random.choice([i for i in list(data) if i.split(".")[0] == 'ADn' and not '.s' in i], nmin, replace = False)
                    },           
            'shuPos':  {
                    'targets'  :   ['x', 'y'],
                    'features'   :   [i for i in list(data) if i.split(".")[0] == 'Pos' and '.s' in i]
                    },          
            'shuADn':  {
                    'targets'  :   ['x', 'y'],
                    'features'   :   [i for i in list(data) if i.split(".")[0] == 'ADn' and '.s' in i]
                    }                    

        }        
# ########################################################################
# # MAIN LOOP FOR SCORE
# ########################################################################

        methods = ['bay_decodage', 'xgb_decodage']
                
        results = {}
        score = {}
        for k in np.sort(combination.keys()):
        # for k in ['allPos']:
            features = combination[k]['features']
            targets = combination[k]['targets']     

            # y_hat = test_decodage(features, targets, methods)            

        
# ########################################################################
# # LEARNING ONE XGB
# ########################################################################    
        X = data[combination['allPos']['features']].values
        Y = data[combination['allPos']['targets']].values
        
        # order is [x, y]    
        nbins_xy = 10
        index = np.arange(nbins_xy*nbins_xy).reshape(nbins_xy,nbins_xy)
        # binning pos
        posbins = np.linspace(0, 1+1e-8, nbins_xy+1)
        xposindex = np.digitize(Y[:,0], posbins).flatten()-1
        yposindex = np.digitize(Y[:,1], posbins).flatten()-1
        # setting class from index
        clas = np.zeros(Y.shape[0])
        for i in xrange(Y.shape[0]): clas[i] = index[xposindex[i],yposindex[i]]

        dtrain = xgb.DMatrix(X, label=clas)        

        params = {'objective': "multi:softprob",
        'eval_metric': "mlogloss", #loglikelihood loss
        'seed': 2925, #for reproducibility
        'silent': 1,
        'learning_rate': 0.05,
        'min_child_weight': 0, 
        'n_estimators': 1000,
        # 'subsample': 0.5,
        'max_depth': 4, 
        'gamma': 0.5,
        'num_class':index.max()+1}
        num_round = 100

        bst = xgb.train(params, dtrain, num_round)

        dtest = xgb.DMatrix(np.atleast_2d(X[0]))
        a = bst.get_dump()
        tmp = []
        # for i in xrange(len(a)):
        ymat1 = bst.predict(dtest, ntree_limit = 1)
        ymat2 = bst.predict(dtest, ntree_limit = 2)
        # tmp.append(ymat1[0])
        # tmp = np.array(tmp)
        ymat = bst.predict(dtrain)

        clas_to_tree = find_featuresconditions(bst.get_dump(), params['num_class'], num_round)
        conditions = find_conditions(clas_to_tree, X)

        n_features = X.shape[1]
        bounds = {}
        for c in conditions.keys():
            tmp = np.zeros((2,n_features)) # column = (lower, upper)            
            for f in conditions[c].iterkeys():
                feat_ind = int(f[1:])
                ind = np.where(conditions[c][f][:,1] == np.min(conditions[c][f][:,1]))[0]
                tmp[0,feat_ind] = np.min(conditions[c][f][ind,0])
                tmp[1,feat_ind] = np.max(conditions[c][f][ind,0])
            bounds[c] = tmp

        from pylab import *
        figure()
        for c in bounds.iterkeys():
            fill_between(np.arange(len(bounds[c][0])), bounds[c][0], bounds[c][1], alpha = 0.1)
        show()

        sys.exit()
        
        # return x[np.argmax(ymat,1)]        


        #     sys.exit()
        #     from pylab import *
        #     figure()
        #     plot(data['ang'].values, label = 'real')
        #     plot(y_hat[:,0], label = 'pred')
        #     legend()

        #     figure()
        #     plot(data['x'].values, data['y'].values, label = 'real')
        #     plot(y_hat[:,1], y_hat[:,2], label = 'pred')
        #     legend()

        #     show()

        #     sys.exit()
        #     y_hat = test_decodage(features, targets, methods)            
        #     results[k] = y_hat
        #     score[k] = {}
        #     y = data['ang'].values
        #     for m in methods:
        #         tmp = np.abs(y_hat[m]-y)
        #         tmp[tmp>np.pi] = 2*np.pi - tmp[tmp>np.pi]
        #         score[k][m] = np.sum(tmp)

            
        # final_data[ses] = {}
        # final_data[ses]['wake'] = {'score':score, 'output':results}

        sys.exit()