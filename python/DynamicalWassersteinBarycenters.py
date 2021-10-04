# -*- coding: utf-8 -*-
import torch
import scipy.io as scio
import numpy as np
import DiffWassersteinLib as dwl
import SimplexRandomWalkUtils as util
import PSD_RiemannianOptimization as psd
import matplotlib.pyplot as plt
import DataSetParameters as dsp
import CostBarycenterLib as cbl
import argparse
import os, sys

class TimeSeriesCost():
    def __init__(self, params, y, pi, mu0, cov0, geoMean, geoCov, x0=None, gamma=None, A_Beta=None, B_Beta=None, A_Beta0=None, B_Beta0=None, w_Beta=None):
        self.p = params
        self.y = y
        self.pi = pi
        self.x0 = x0
        self.mu0 = mu0
        self.cov0 = cov0
        self.gamma = gamma
        self.geoMean = geoMean
        self.geoCov = geoCov
        self.A_Beta = A_Beta
        self.B_Beta = B_Beta
        self.A_Beta0 = A_Beta0
        self.B_Beta0 = B_Beta0
        self.w_Beta = w_Beta
        self.computeEval = 0
        
    def cost(self, gamma, x0, mu, covP, A0_beta, B0_beta, A_beta, B_beta, w_Beta):
        # compute time series state given step parameter
        X2 = util.StateEvolutionDynamics(self.p, x0, self.pi, gamma)
        
        cov = []
        for i in range(self.p.K):
            cov.append(covP[i])
        
        # Update the current mean and variance parameters
        clustMean = []
        clustCov = []
        for i in range(self.p.K):
            clustMean.append(mu[i])
            clustCov.append(cov[i])
        
        # Now compute barycentric distributional parameters and pdf for each observation
        obsCost = []
        baryMeans=[]
        baryCovs=[]
        obsEval = []
        for i in range(self.p.T):
            if (self.geoCov == 'GMM'): # The gmm models as a mixture
                obsCost.append(util.GaussGmm_WassDist(y[i][0], y[i][1], mu, cov, X2[i])**2)
                # This monte carlo simulation is costly, only update the evaluation when writing debug file
                if (self.computeEval==1):
                    obsEval.append(util.GaussGmm_WassDist_MonteCarlo(y[i][0].detach().numpy(), y[i][1].detach().numpy(), 
                                                                 mu, cov, X2[i].detach().numpy()))
                else:
                    obsEval.append(0)
                
            else:  # all other models output a single gaussian
                baryMeans.append(self.geoMean.barycenter(torch.stack(clustMean),X2[i]))
                baryCovs.append(self.geoCov.barycenter(torch.stack(clustCov), X2[i]))
                
                # Loss is Wasserstein distance to barcenter distribution
                obsCost.append(self.geoMean.distance(y[i][0], baryMeans[i])**2 + 
                                 cbl.WassersteinPSD.dist(y[i][1], baryCovs[i])**2)
                obsEval.append(obsCost[i].detach().numpy()) # eval same as cost

        # Compute loss for step parameters(gamma)
        log_pGamma = util.LogGammaLiklihoodBimodalAB(p, A_Beta0, B_Beta0, A_Beta, B_Beta, w_Beta, gamma)
        
        # Pure state parameter prior
        log_pTheta = []
        for i in range(self.p.K):
            log_pTheta.append(-torch.sqrt(2*torch.tensor(np.pi))*self.p.cluster_sig +  -1/(2*self.p.cluster_sig**2)*util.GaussWassDistance(self.mu0, self.cov0, mu[i], cov[i]) )
                
        lossFunc = self.p.regObs * (torch.sum(torch.stack(obsCost))) - torch.tensor(self.p.T)*(torch.sum(torch.stack(log_pTheta))) - (torch.sum(log_pGamma)) 
        
        if (torch.isnan(lossFunc)):
            print('stop')
            raise Exception("bad value")
        return (lossFunc, X2, torch.stack(obsCost), log_pGamma, torch.stack(log_pTheta), torch.stack(cov), obsEval)
        
    def evaluate(self, listMuSig):
        n = int(len(listMuSig)/2)
        return self.cost(self.gamma, self.x0, [torch.tensor(x) for x in listMuSig[:n]], [torch.tensor(x) for x in listMuSig[n:]], self.A_Beta0, self.B_Beta0, self.A_Beta, self.B_Beta, self.w_Beta)[0].detach().numpy()
        
if __name__=="__main__":
    #Initialize Data
    dtype = torch.float
    device = torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataSet", type=str, default=None)
    parser.add_argument("dataFile", type=str, default=None)
    parser.add_argument("debugFolder", type=str, default=None)
    parser.add_argument("geoCov", type=str, default=None)
    parser.add_argument("--ParamTest", dest="ParamTest", default=0, type=int) # Used in ParamTest
    parser.add_argument("--lambda", dest="lam", type=float) # Used in ParamTest
    parser.add_argument("--s", dest="s", type=float) # Used in ParamTest
    ParamTest=0
    
    try:
        print("Reading Arguments from Command Line")
        args = parser.parse_args()
        p = dsp.GetDataParameters(args.dataSet)
        p.update({'dataFile':args.dataFile, 'debugFolder':args.debugFolder, 'geoCov':args.geoCov, 'paramFile':args.geoCov+'_params.txt'})
        if (args.ParamTest ==1):
            p.update({'regObs':args.lam, 'cluster_sig':args.s})
            if (args.ParamTest==1):
                ParamTest=1
        else:
            print("Not Running ParamTest test")
                    
        print('DataFile: ' + p.dataFile)
        print('DebugFolder: ' + p.debugFolder)
        print('geoCov: ' + p.geoCov)
    except:
        print("Instead using default Params: ", sys.exc_info()[0])
        # Load dataset. Here we need defined y = observationsde
        dataSet = 'MSR_Batch'
        p = dsp.GetDataParameters(dataSet)
    
    datO = scio.loadmat(p.dataFile)[p.dataVariable].astype(float)
    dat = util.WindowData(datO, p.window, p.stride, p.offset)
    (T, dump, dim) = dat.shape

    try: # If the file contains a K value, use it.
        K = np.squeeze(scio.loadmat(p.dataFile)['K'].astype(int))
        p.update({'K':K})
    except:
        print('Using default K Value')

    # Update and print parameters
    p.update({'T':T, 'dim':dim})
    try:
        os.mkdir(p.debugFolder)
    except: 
        print("Folder already exists")
    p.write(f=p.debugFolder + p.paramFile)
    
    # Initialize Model Parameters. 
    gamma = torch.tensor(np.zeros((p.T,p.K))+p.eps, dtype = dtype, requires_grad=True)
    if (ParamTest==0):
        x0 = torch.tensor(np.ones(p.K)/p.K, dtype = dtype, requires_grad=True)
        A_Beta = torch.tensor(np.ones(p.K)*10, dtype = dtype, requires_grad=True)
        B_Beta = torch.tensor(np.ones(p.K)*20, dtype = dtype, requires_grad=True)
        A_Beta0 = torch.tensor(np.ones(p.K)*1.1, dtype = dtype, requires_grad=False)
        B_Beta0 = torch.tensor(np.ones(p.K)*20, dtype = dtype, requires_grad=False)
        w_Beta = torch.tensor(np.ones(p.K)*0.5, dtype=dtype, requires_grad=True)
    else:
        x0 = torch.tensor(np.ones(p.K)/p.K, dtype = dtype, requires_grad=True)
        A_Beta = torch.tensor(np.ones(p.K)*p.alpha, dtype = dtype, requires_grad=True)
        B_Beta = torch.tensor(np.ones(p.K)*p.beta, dtype = dtype, requires_grad=True)
        A_Beta0 = torch.tensor(np.ones(p.K)*1.1, dtype = dtype, requires_grad=False)
        B_Beta0 = torch.tensor(np.ones(p.K)*20, dtype = dtype, requires_grad=False)
        w_Beta = torch.tensor(np.ones(p.K)*0.5, dtype=dtype, requires_grad=True)
        
    meanDat = np.mean(datO, axis=0)
    stdDat = np.std(datO,axis=0)

    if (p.initMethod == 'CPD'): # initialize based on K CPD Fit. 
        cpd = util.CPD_WM1(datO, p.window*2)
        (muO, sigO) = util.CPD_Init(datO, cpd, max(cpd)*0.2, p.K)
    elif (p.initMethod == 'label'): # Alternately initialize based on labels
        L = np.squeeze(scio.loadmat(p.dataFile)['L'].astype(int))
        (muO, sigO) = util.label_Init(datO, L)
        cpd=[]

    (muP, sigP) = util.FitMuSig(datO, p.K) # Regularize based on distance to (mean of gmm parameters, average eValue of gmm Covariances)
    covDat = dwl.CovarianceBarycenter(torch.tensor(sigO, dtype=dtype), torch.ones(p.K)/p.K, torch.eye(p.dim), nIter=10).detach().numpy()

    p.muPrior=torch.tensor(muP, dtype=dtype, requires_grad=False)
    p.covPrior=torch.tensor(sigP, dtype=dtype, requires_grad=False)
        
    # Compue empirical means and covariances of the input time sereis
    y=[]
    for i in range(p.T):
        obsMean = np.mean(dat[i], axis=0)
        obsCov = 1/(p.window-1)*np.matmul((dat[i]-obsMean).T, dat[i]-obsMean)
        minDiag=1e-2
        for j in range(p.dim): #just to make sure we are not singular
            obsCov[j,j]=max(minDiag,obsCov[j,j])
        y.append((torch.tensor(obsMean, dtype=dtype, requires_grad=False), torch.tensor(obsCov, dtype=dtype, requires_grad=False)))
    pi = torch.tensor(np.ones(p.K)/p.K, dtype=dtype, requires_grad=False) # this needs to sum to 1
    
    # Setup params for manifold optimization 
    man = []
    mu = torch.tensor(muO, dtype=dtype, requires_grad=True)
    for i in range(p.K): # Euclidean manifold for Means
        man.append(psd.Euclidean())
        geoMean = cbl.Euclidean()
    for i in range(p.K): # WB manifold for cov matrices
        man.append(psd.WassersteinBuresPSDManifold())
    covP = torch.tensor(sigO, dtype=dtype, requires_grad=True)

    if (p.geoCov=='Wass'):
        geoCov = cbl.WassersteinPSD(torch.eye(p.dim))
    elif (p.geoCov=='GMM'):
        geoCov = 'GMM'
           

    # Setup optimization params            
    SManifold = psd.Product(man)
    optimGamma = torch.optim.Adam([gamma, x0], lr=p.lr_Gamma)
    optimAB = torch.optim.Adam([A_Beta, B_Beta, w_Beta], lr=p.lr_Gamma)
    init=0
    swap = 'Gamma' # Coordinate descent
    runningCount=0
    costFunc = TimeSeriesCost(p, y, pi, p.muPrior, p.covPrior, geoMean, geoCov)
    history=[]
    evalHistory=[]
    cyclicPoints=[]
    # Start Optimization 
    for t in range(p.nOptimStep):
        # Since the monte carlo simulations for GMM evaluation takes a long time, set a flag to indicate we are running debug
        if (t % p.printInterval == 0): 
            costFunc.computeEval = 1
            (lossFunc, X2, obsCost, log_pGamma, log_pTheta, cov, obsEval) = costFunc.cost(gamma, x0, mu, covP, A_Beta0, B_Beta0, A_Beta, B_Beta, w_Beta)
            evalHistory.append(np.sum(obsEval))
            costFunc.computeEval = 0
        else:
            (lossFunc, X2, obsCost, log_pGamma, log_pTheta, cov, obsEval) = costFunc.cost(gamma, x0, mu, covP, A_Beta0, B_Beta0, A_Beta, B_Beta, w_Beta)
        
        # Backward pass: compute gradient of the loss with respect to model
        lossFunc.backward()
        history.append(lossFunc.detach().numpy())
        
        # Log results
        print(lossFunc)
        if (p.logFile is not None):
            print(lossFunc, file=p.logFile)
             
        # Update for gamma
        if (swap == 'Gamma'):    
            # Update for Gammas
            optimGamma.step()
    
            # Kind of a hack, but we clamp Gamma to [0,1] and apply boundary conditions to x0 outside of the gradient update
            gamma.data = gamma.clamp(min=p.eps, max=1-p.eps)
            x0.data = x0.clamp(min=p.eps)
            x0.data = x0.data/torch.sum(x0)
        elif (swap == 'AB'):
            optimAB.step()
            A_Beta.data = A_Beta.clamp(min=1.1)
            for k in range(p.K):
                B_Beta.data[k] = B_Beta[k].clamp(min=1.1, max=A_Beta[k].detach().numpy()*(1-0.15)/0.15) # mean of Beta distribution >= 0.1
            w_Beta.data = w_Beta.clamp(min=0.01, max=0.99)
            
        else: # Line Search update for means
            xt = [mu[i].detach().numpy() for i in range(p.K)] + [covP[i].detach().numpy() for i in range(p.K)]
            xt_nGrad  = [-mu.grad[i].detach().numpy() for i in range(p.K)] + [-covP.grad[i].detach().numpy() for i in range(p.K)]
            
            if (p.cyclic_MuSig): 
                for i in range(p.K*2): 
                    if (i % p.K != runningCount % (p.K)): #Optimize one mean at a time
                        xt_nGrad[i] = xt_nGrad[i]*0
                    
            riemannianGradient = SManifold.euc_to_riemannian_gradient(xt, xt_nGrad)
            
            optimMean = psd.LineSearchSimple(SManifold, TimeSeriesCost(p, y, pi.detach(), p.muPrior, p.covPrior, geoMean, geoCov, x0=x0.detach(), gamma=gamma.detach(), A_Beta0=A_Beta0.detach(), B_Beta0=B_Beta0.detach(), A_Beta=A_Beta.detach(), B_Beta=B_Beta.detach(), w_Beta=w_Beta.detach()), suff_decrease=1e-10, maxIter=20, init_alpha=p.lr_Cluster)
            update = optimMean.search(xt, [x for x in riemannianGradient])
            mu.data=torch.tensor(update[:p.K], dtype=dtype)
            covP.data=torch.tensor(update[p.K:], dtype=dtype)

        # Reset Gradient Computation        
        x0.grad.data.zero_()
        gamma.grad.data.zero_()
        mu.grad.data.zero_()
        covP.grad.data.zero_()
        A_Beta.grad.data.zero_()
        B_Beta.grad.data.zero_()
        w_Beta.grad.data.zero_()
                 
        # Coordinate Descent criterea
        runningCount = runningCount+1
        if ((runningCount > (p.K*2) and history[-p.K]-history[-1] < p.cyclicThresh*(p.K)) or runningCount > p.cyclicIterMax):
            if (swap == 'Gamma'):
                swap = "AB"
                print("Swapping from Gamma to AB optim after " + str(runningCount) + " steps")
            elif (swap == 'AB'):
                swap = "Cluster"
                print("Swapping from AB to Cluster optim after " + str(runningCount) + " steps")
            else:
                swap = 'Gamma'    
                print("Swapping from Cluster to Gamma optim after " + str(runningCount) + " steps")
            runningCount=0
            cyclicPoints.append(t)
        
        # Save Data
        if (t % p.printInterval == 0 and not torch.isnan(lossFunc)): # Print Interval
            print("Save Debug " + p.debugFolder + p.geoCov +"_"+  p.outputFile )
            scio.savemat(p.debugFolder + p.geoCov +"_"+ p.outputFile, mdict={'t':t, 
                                              'dat':dat,
                                              'pi':pi.detach().numpy(),
                                              'datO':datO,
                                              'meanDat':meanDat,
                                              'covDat':covDat,
                                              'meanP':muP,
                                              'sigP':sigP,
                                              'cpd':cpd,
                                              'cyclicPoints':cyclicPoints,
                                              'log_pGamma':log_pGamma.detach().numpy(),
                                              'log_pTheta':log_pTheta.detach().numpy(),
                                              'X':X2.detach().numpy(),
                                              'obsCost':obsCost.detach().numpy(),
                                              'obsEval':obsEval,
                                              'evalHistory':evalHistory,
                                              'history':history,
                                              'window':p.window,
                                              'stride':p.stride,
                                              'x0':x0.detach().numpy(), 
                                              'gamma':gamma.detach().numpy(), 
                                              'mu':mu.detach().numpy(), 
                                              'covP':covP.detach().numpy(), 
                                              'covv':cov.detach().numpy(), 
                                              'muO':muO,
                                              'covO':sigO,
                                              'A_Beta0':A_Beta0.detach().numpy(),
                                              'B_Beta0':B_Beta0.detach().numpy(),
                                              'A_Beta':A_Beta.detach().numpy(),
                                              'B_Beta':B_Beta.detach().numpy(),
                                              'w_Beta':w_Beta.detach().numpy()})