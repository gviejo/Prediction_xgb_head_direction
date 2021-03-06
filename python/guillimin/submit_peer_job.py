#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""                                                                                   
submit_job.py                                                                         
                                                                                      
to launch sferes job on cluster                                                       
                                                                                      
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.                              
"""

import sys,os
from optparse import OptionParser
import pandas as pd
import scipy.io
import numpy as np
import cPickle as pickle
from time import sleep

########################################################################
# CREATE DIRECTORY RESULTS
########################################################################


#####################################################################
# SESSION LOADING
#####################################################################
all_sessions = {}
# for ep in ['wake', 'rem', 'sws']:
for ep in ['wake']:    
    all_sessions[ep] = os.listdir(os.path.expanduser("~/sessions/"+ep+"/"))
    for ses in all_sessions[ep]:        
        adrien_data = scipy.io.loadmat("/home/viejo/sessions/"+ep+"/"+ses)
        session = ses.split(".")[1]
        adn = adrien_data['ADn'].shape[1]
        pos = adrien_data['Pos'].shape[1]
        
        if adn >= 7 and pos >= 7:   
########################################################################
# GENERATE BASH SCRIPTS                                                               
########################################################################        
            filename = "submit_"+ep+"_"+ses.split(".")[1]+".sh"
            f = open(filename, "w")
            f.writelines("#!/bin/bash\n")
            f.writelines("#PBS -l nodes=1:ppn=16\n")
            f.writelines("#PBS -l walltime=01:00:00\n")
            f.writelines("#PBS -A exm-690-aa\n")
            f.writelines("#PBS -j oe\n")
            f.writelines("#PBS -N peer_"+ep+"_"+ses.split(".")[1]+"\n")
            f.writelines("\n")                                                  
            f.writelines("python /home/viejo/Prediction_ML_GLM/python/guillimin/cluster_peer_pr2_sessions.py -e "+ep+" -s "+ses+"\n")
            f.close()
            #-----------------------------------                
            # ------------------------------------                                                                                            
            # SUBMIT                                                                                                                          
            # ------------------------------------                                                                                            
            os.system("chmod +x "+filename)
            os.system("qsub "+filename)
            os.system("rm "+filename)
            print session, filename, adn, pos
            sleep(10)