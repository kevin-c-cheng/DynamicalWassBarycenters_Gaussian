# -*- coding: utf-8 -*-
import sys

class TimeSeriesParams():
    def __init__(self, params={}):
        self.dParams={}
        # set default
        
         # Run Config Parameters
        self.dParams['logFile'] = None
        self.dParams['paramFile'] = 'params.txt'
        self.dParams['debugFolder'] = None
        self.dParams['outputFile'] = 'out.mat'
        self.dParams['dataName'] = 'test'
        self.dParams['dataFile'] = None
        self.dParams['dataVariable'] = 'y'
        self.dParams['initMethod'] = 'GMM'
        self.dParams['inputType'] = 'Sample'
        
         # Model Parameters
        self.dParams['alpha'] = 1.1
        self.dParams['beta'] = 10
        self.dParams['window'] = 1
        self.dParams['stride'] = 1
        self.dParams['offset'] = 0
        self.dParams['eps'] = 1e-6
        self.dParams['geoMean'] = "Euc" # Geometry to evaluate and optimize the Mean
        self.dParams['geoCov'] = "Wass" # Geometry to evaluate and optimize the Covariance
        self.dParams['regObs'] = 1
        
        # Gaussian Model Specific Parameters
        self.dParams['cluster_sig'] = 10 # Prior on Wasserstein distance to Cluster center: Gaussian standard deviation 

       # Optimization Parameters
        self.dParams['nOptimStep'] = 50000 
        self.dParams['lr_Gamma'] = 2e-3
        self.dParams['lr_Cluster'] = 1e-1
        self.dParams['nCyclic'] = 200
        self.dParams['printInterval'] = 500
        self.dParams['cyclicThresh'] = 1
        self.dParams['cyclicIterMax'] = 1000
        self.dParams['optimMethod_Cluster'] = "LineSearch"
        self.dParams['cyclic_MuSig'] = True
        
        # Internal Parameters
        self.dParams['T'] = 0
        self.dParams['dim'] = 0
        self.dParams['K'] = 0
        
        # Update and Save
        self.update(params)
        self.save()
        
    def save(self):
        # Store parameters"
         # Run Config Parameters
        self.logFile = self.dParams['logFile']
        self.paramFile = self.dParams['paramFile'] 
        self.debugFolder = self.dParams['debugFolder']
        self.outputFile = self.dParams['outputFile']
        self.dataName = self.dParams['dataName']
        self.dataFile = self.dParams['dataFile']
        self.dataVariable = self.dParams['dataVariable']
        self.initMethod = self.dParams['initMethod']
        assert (self.initMethod == 'CPD' or self.initMethod == 'GMM' or self.initMethod == 'label')
        self.inputType = self.dParams['inputType']
        assert (self.inputType== 'Sample' or self.inputType == 'Gaussian')

         # Model Parameters
        self.alpha = self.dParams['alpha']
        self.beta = self.dParams['beta']
        self.window = self.dParams['window']
        self.stride = self.dParams['stride']
        self.offset = self.dParams['offset']
        self.eps = self.dParams['eps']
        self.geoMean = self.dParams['geoMean'] 
        assert (self.geoMean == 'Euc')
        self.geoCov = self.dParams['geoCov']
        assert (self.geoCov == 'Wass' or self.geoCov == 'Euc' or self.geoCov == 'Hel' or self.geoCov == 'GMM')
        self.regObs = self.dParams['regObs']

        # Gaussian Model Specific Parameters
        self.cluster_sig = self.dParams['cluster_sig']

         # Optimization Parameters
        self.nOptimStep = self.dParams['nOptimStep']
        self.lr_Gamma = self.dParams['lr_Gamma']
        self.lr_Cluster = self.dParams['lr_Cluster']
        self.nCyclic = self.dParams['nCyclic']
        self.printInterval = self.dParams['printInterval']
        self.cyclicThresh = self.dParams['cyclicThresh']
        self.cyclicIterMax = self.dParams['cyclicIterMax']

        self.optimMethod_Cluster = self.dParams['optimMethod_Cluster']
        assert (self.optimMethod_Cluster == 'GradientDescent' or self.optimMethod_Cluster == 'LineSearch')
        self.cyclic_MuSig = self.dParams['cyclic_MuSig']

        
         # Internal Parameters
        self.T = self.dParams['T']
        self.dim = self.dParams['dim']
        self.K = self.dParams['K']

    def update(self, params):
        # Update based on inputs
        for k in params.keys():
            self.dParams[k] = params[k]
        self.save()

        
    def write(self, f=sys.stdout):
        if (f is None):
            f = sys.stdout
        else:
            f = open(f, 'w')
        print("Parameter Dump", file=f)
        for k in self.dParams.keys():
            print("  " + k + ": \t" + str(self.dParams[k]), file = f)
        print(" ", file=f)

        if (f != sys.stdout):
            f.close()
