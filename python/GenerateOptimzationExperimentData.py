# -*- coding: utf-8 -*-
import torch
import scipy.io as scio
import numpy as np
import scipy.stats as scistats
import PSD_RiemannianOptimization as psd

dim = [2,5,10,20,50,70,100,150,200]
nSamp = 100
distS = 2
distM = 1
nIter = 10
folderOutput = "../data/SimulatedOptimizationData_2K"

WBManifold = psd.WassersteinBuresPSDManifold()
EucManifold = psd.Euclidean()

np.random.seed(0)
for d in dim:
    run=True
    for nI in range (nIter):
        count = 0
        while (run):
            # Generate Random PSD matrix
            eVal = np.random.rand(d)+0.5
            AS = scistats.random_correlation.rvs(eVal* (d/sum(eVal)))*(sum(eVal)/d)
            
            # Generate Random Symmetric Matrix Tangent Vector
            tanS = np.random.rand(d,d)
            tanS = 0.5 * (tanS + tanS.transpose())
        
            # The second matrix is the picked from intial point and the tangent vector with a fixed distance
            tanS_Mag = np.sqrt(WBManifold.riemannian(AS, tanS, tanS))
            BS = WBManifold.exp(AS, tanS*(distS/tanS_Mag))
        
            # Generate Random Mean
            Am = np.random.rand(d,1)
            
            # Generate Random Mean Tangent Vector
            tanM = np.random.rand(d,1)    
            
            # The second mean is given from the initial point and tangent vector
            tanM_Mag = np.sqrt(EucManifold.riemannian(Am, tanM, tanM))
            Bm = EucManifold.exp(Am, tanM*(distM/tanM_Mag))
            
            # Check Results
            (A_eig, A_evec) = np.linalg.eig(AS)
            Acond = max(A_eig)/min(A_eig)
            (B_eig, B_evec) = np.linalg.eig(BS)
            Bcond = max(B_eig)/min(B_eig)
             
            if (min(A_eig)< 1e-10 or min(B_eig < 1e-10) or Acond > 1e10 or Bcond > 1e10):
                run=True
            else:
                run=False
                count = count+1
        
        # Verify Output
        print("Dim: ", d)
        print("Dist PSD:", np.sqrt(WBManifold.distance(AS, BS)))
        print("Dist mean:", np.sqrt(EucManifold.distance(Am, Bm)))
        print("A condition #: ", Acond, " B condition #: ", Bcond)
        print("tries: ", count)
        print("")
        
        # Generate intermediate interpolated distributions
        tAll = np.linspace(0,1,nSamp)
        datM = np.zeros((nSamp, d,1))
        datS = np.zeros((nSamp, d,d))
        count = 0
        n = d*20
        for t in tAll:
            tM = Am * (1-t) + Bm * t
            tS = WBManifold.geodesic(AS, BS, t)
            run = True
            while (run):
                samps = np.random.multivariate_normal(np.squeeze(tM), tS, n)
                if (min(np.linalg.eig(np.cov(np.transpose(samps)))[0])>0):
                    run=False
            datM[count,:,:] = np.expand_dims(np.mean(samps,0),1)
            datS[count,:,:] = np.cov(np.transpose(samps))
            count = count + 1
        
        d1=[]
        d2=[]
        for i in range(nSamp):
            d1.append(WBManifold.distance(AS, datS[i]))
            d2.append(WBManifold.distance(BS, datS[i]))
        
        mus = np.zeros((2,d))
        mus[0,:] = np.squeeze(Am)
        mus[1,:] = np.squeeze(Bm)
        covs = np.zeros((2,d,d))
        covs[0,:] = AS
        covs[1,:] = BS
        X = np.zeros((nSamp,2))
        X[:,0] = (1-tAll)
        X[:,1] = tAll
            
        # Save the Data
        scio.savemat(folderOutput + '/dim_' + str(d) + '_'+str(nI)+ '.mat', mdict={'datM':datM, 'datS':datS, 'X':X, 'mus':mus, 'covs':covs, 'dim':d, 'd1':d1, 'd2':d2})
    