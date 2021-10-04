# -*- coding: utf-8 -*-

import os
import glob
import TimeSeriesParams as TSP


def GetDataParameters(dataSet, sel = None):
    p = TSP.TimeSeriesParams()
    if (dataSet == "BeepTest"):
        params={
            'dataName':'Beep Test' }
        params['dataFile']='../data/BeepTestData/beep_3103_pre_Trunc.mat'
        params['debugFolder'] = "..//debug/BeepTestDebug/"
        params['dataVariable'] = 'Y'
        params['K'] = 2
        params['window'] = 100
        params['stride'] = 25
#        params['stride'] = 50

        params['cluster_sig'] = 1.0
        params['alpha'] = 1.1
        params['beta'] =  3
        params['regObs'] = 100
        params['initMethod'] = 'CPD'

        params['cyclicIterMax'] = 1000
        params['cyclicThresh'] = 0.1
    elif (dataSet == "MSR_Batch"):
        # This data is normalized for its mean and standard deviation.
        params={
            'dataName':'MSR'}
        params['dataFile']='../data/MSR_Data/subj001_1.mat'
        params['debugFolder'] = "../debug/MSR_Batch/"
        params['dataVariable'] = 'Y'

        params['window'] = 250
        params['stride'] = 125

        params['cluster_sig'] = 1.0 
        
        params['alpha'] = 1.1
        params['beta'] =  3
        params['regObs'] = 100 #10
        params['initMethod'] = 'CPD'

        params['cyclicIterMax'] = 1000
        params['cyclicThresh'] = 0.05
    elif (dataSet == "MSR_BatchGT"):
        # This data is normalized for its mean and standard deviation.
        params={
            'dataName':'MSR'}
        params['dataFile']='../data/MSR_Data/subj001_1.mat'
        params['debugFolder'] = "../debug/MSR_Batch/"
        params['dataVariable'] = 'Y'
#        params['K'] = 10
        params['K'] = 4 #7
        params['window'] = 250 
        params['stride'] = 125

        params['cluster_sig'] = 1.0
        
        params['alpha'] = 1.1
        params['beta'] =  3
        params['regObs'] = 100
        params['regObs'] = 10

        params['initMethod'] = 'label'

        params['cyclicIterMax'] = 1000
        params['cyclicThresh'] = 0.05

    elif (dataSet == "ParamTest"):
        # This data is normalized for its mean and standard deviation.
        params={
            'dataName':'paramTest'}
        params['dataFile']='../data/ParamTest/subj001_1.mat'
        params['debugFolder'] = "../data/debug/ParamTest/"
        params['dataVariable'] = 'Y'

        params['window'] = 250 
        params['stride'] = 125

        params['K'] = 2
        params['alpha'] = 1.1
        params['beta'] =  3
        
        params['initMethod'] = 'CPD'

        params['cyclicIterMax'] = 1000
        params['cyclicThresh'] = 0.05
    p.update(params)
    return p

