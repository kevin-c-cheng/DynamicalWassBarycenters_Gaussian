# -*- coding: utf-8 -*-
import torch
import numpy as np
import argparse
import DiffWassersteinLib as dwl
import PSD_RiemannianOptimization as psd
import scipy.io as scio
import time


class Cost():
    def __init__(self, X, T, K, mode):
        self.X = X
        self.T = T
        self.K = K
        self.mode = mode

    def cost(self, stateMean, stateCov):
        X2 = self.X
        # Get the covariance matrix
        if (self.mode == "WB"):
            stateCov2 = stateCov
        else: # Parameterize by LL
            stateCov2 = []
            for i in range(K):
                tmpM = torch.zeros((d, d))
                tmpM[tril_indices[0], tril_indices[1]] = stateCov[i]
                stateCov2.append(torch.matmul(tmpM, tmpM.T))        
        baryMeans=[]
        baryCovs=[]
        cost=[]
        for i in range(self.T):
            if (self.K==3):
                baryMeans.append(X2[i][0]*stateMean[0] + X2[i][1]*stateMean[1] + X2[i][2]*stateMean[2])
            elif (self.K==2):
                baryMeans.append(X2[i][0]*stateMean[0] + X2[i][1]*stateMean[1])
                
            baryCovs.append(dwl.CovarianceBarycenter(torch.stack(stateCov2), X2[i], torch.eye(d), nIter=4))            
            cost.append(torch.pow(psd.Euclidean.dist(torch.tensor(datM[i], dtype=torch.float, requires_grad=False), baryMeans[i]),2) +
                            psd.WassersteinBuresPSDManifold.dist(torch.tensor(datS[i], dtype=torch.float, requires_grad=False), baryCovs[i]))
            totalCost = torch.sum(torch.stack(cost))
        return totalCost
    
    def evaluate(self, parameters):
        K = int(len(parameters)/2)
        return self.cost([torch.tensor(x) for x in parameters[:K]], [torch.tensor(x) for x in parameters[K:]]).detach().numpy()


if __name__=="__main__":
    torch.manual_seed(0)
    np.random.seed(0)
   
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("fileIn", type=str, default=None)
        parser.add_argument("mode", type=str, default=None)
        parser.add_argument("fileOut", type=str, default=None)
        args = parser.parse_args()
        fileIn = args.fileIn
        fileOut = args.fileOut
        mode = args.mode
    except:
        fileIn = "../data/SimulatedOptimizationData_2K/dim_5_0.mat"
        fileOut = "dump.mat"
        mode = "WB" #"WB"
        
    print(fileIn)
    print(fileOut)
    print(mode)

    datM = scio.loadmat(fileIn)["datM"].astype(float)
    datS = scio.loadmat(fileIn)["datS"].astype(float)
    mus = scio.loadmat(fileIn)["mus"].astype(float)
    covs = scio.loadmat(fileIn)["covs"].astype(float)
    X = scio.loadmat(fileIn)["X"].astype(float)
    d = int(scio.loadmat(fileIn)["dim"].astype(float).squeeze())
    T = X.shape[0]
    K = X.shape[1]
    
    initMean = datM[int(T/2)]
    initS = datS[int(T/2)]
    
    stateMean = [torch.tensor(initMean, dtype=torch.float, requires_grad=True) for i in range(K)]
    if (mode == "WB"): # Wasserstein-Bures
        WB = psd.WassersteinBuresPSDManifold()
        stateCov = [torch.tensor(initS, dtype=torch.float, requires_grad=True) for i in range(K)]
    else: # Parameterize by LL
        tril_indices = torch.tril_indices(row=d, col=d, offset=0)
        tmp = np.linalg.cholesky(initS)[np.tril_indices(d)]
        stateCov = [torch.tensor(tmp, dtype=torch.float, requires_grad=True) for i in range(K)]


    X2 = torch.tensor(X, dtype=torch.float, requires_grad = False)

    nIter=250
    historyCost = []
    historyTime = []
    historyEval = []
    gradientStep = 1e-2
    t0 = time.time()
    for t in range(nIter):        
        # Compute the cost / distance
        CFunc = Cost(X2, T, K, mode)
        totalCost = CFunc.cost(stateMean, stateCov)
        totalCost.backward()

        # Gradient Descent: Optimize the Parameters 
        if (0): #'GradientDescent'):
            if (mode == "WB"):
                for i in range(K):
                    stateMean[i].data = stateMean[i] - gradientStep * stateMean[i].grad
                    tan = WB.euc_to_riemannian_gradient(stateCov[i].detach().numpy(), -stateCov[i].grad.detach().numpy())
                    stateCov[i].data = torch.tensor(WB.exp(stateCov[i].detach().numpy(), gradientStep*tan), dtype=torch.float)
            else:
                for i in range(K):
                    stateMean[i].data = stateMean[i] - gradientStep * stateMean[i].grad
                    stateCov[i].data = stateCov[i] - gradientStep * stateCov[i].grad
        elif (1): #'LineSearch'): # Line Search
            xt = [stateMean[i].detach().numpy() for i in range(K)] + [stateCov[i].detach().numpy() for i in range(K)]
            xt_nGrad  = [-stateMean[i].grad.detach().numpy() for i in range(K)] + [-stateCov[i].grad.detach().numpy() for i in range(K)]

            if (mode == "WB"):
                man = []
                for i in range(K): # means
                    man.append(psd.Euclidean())
                for i in range(K): # S
                    man.append(psd.WassersteinBuresPSDManifold())
            else: # "LL"
                man = []
                for i in range(K): # means
                    man.append(psd.Euclidean())
                for i in range(K): # S
                    man.append(psd.Euclidean())
                    
            SManifold = psd.Product(man)

            optim = psd.LineSearchSimple(SManifold, CFunc, suff_decrease=1e-5, maxIter=20, invalidCost=0)
            riemannianGradient = SManifold.euc_to_riemannian_gradient(xt, xt_nGrad)
            update = optim.search(xt, [x for x in riemannianGradient])

            for i in range(K):
                stateMean[i].data = torch.tensor(update[i], dtype=torch.float)
                stateCov[i].data = torch.tensor(update[i+K], dtype=torch.float)

        
        # Eval = total wass distance to end points
        evalOut=0
        for i in range(K):
            if (mode == "WB"):
                stateCov2 = stateCov
            else: # Parameterize by LL
                stateCov2 = []
                for i in range(K):
                    tmpM = torch.zeros((d, d))
                    tmpM[tril_indices[0], tril_indices[1]] = stateCov[i]
                    stateCov2.append(torch.matmul(tmpM, tmpM.T)) 
            evalOut += psd.Euclidean.dist(mus[i], stateMean[i].detach().numpy())**2
            evalOut += psd.WassersteinBuresPSDManifold.dist(covs[i], stateCov2[i].detach().numpy())
        
        # Log and update
        historyCost.append(totalCost.detach().numpy())
        historyTime.append(time.time()-t0)
        historyEval.append(evalOut)
        for i in range(K):
            stateMean[i].grad.data.zero_()
            stateCov[i].grad.data.zero_()
            
        stateMeanOut=[x.detach().numpy() for x in stateMean]
        stateCovOut=[x.detach().numpy() for x in stateCov]
        print(t, ": " , totalCost.detach().numpy())
        scio.savemat(fileOut, mdict={'historyCost':historyCost, 'historyTime':historyTime, 'historyEval':historyEval, 'stateMean':stateMeanOut, 'stateCov':stateCovOut, 'datM':datM, 'datS':datS})
    