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
folderOutput = "../data/SimulatedOptimizationData_3K"

WBManifold = psd.WassersteinBuresPSDManifold()
EucManifold = psd.Euclidean()

np.random.seed(0)
for d in dim:
    run=True
    for nI in range (nIter):
        count = 0
        while (run):
            # Generate Random PSD matrix
            fail=1
            while(fail==1):
                try:
                    eVal = np.random.rand(d)+0.5
                    AS = scistats.random_correlation.rvs(eVal* (d/sum(eVal)))*(sum(eVal)/d)
                    fail=0
                except:
                    fail=1
                
            
            # Generate Random Symmetric Matrix Tangent Vector
            tanS = np.random.rand(d,d)
            tanS = 0.5 * (tanS + tanS.transpose())

            tanS2 = np.random.rand(d,d)
            tanS2 = 0.5 * (tanS2 + tanS2.transpose())
        
            # The second matrix is the picked from intial point and the tangent vector with a fixed distance
            tanS_Mag = np.sqrt(WBManifold.riemannian(AS, tanS, tanS))
            BS = WBManifold.exp(AS, tanS*(distS/tanS_Mag))
                  
            tanS2_Mag = np.sqrt(WBManifold.riemannian(BS, tanS2, tanS2))
            CS = WBManifold.exp(BS, tanS2*(distS/tanS2_Mag))
        
            # Generate Random Mean
            Am = np.random.rand(d,1)
            
            # Generate Random Mean Tangent Vector
            tanM = np.random.rand(d,1)    
            tanM2 = np.random.rand(d,1)    
            
            # The second mean is given from the initial point and tangent vector
            tanM_Mag = np.sqrt(EucManifold.riemannian(Am, tanM, tanM))
            Bm = EucManifold.exp(Am, tanM*(distM/tanM_Mag))

            tanM2_Mag = np.sqrt(EucManifold.riemannian(Bm, tanM2, tanM2))
            Cm = EucManifold.exp(Bm, tanM2*(distM/tanM2_Mag))
            
            # Check Results
            (A_eig, A_evec) = np.linalg.eig(AS)
            Acond = max(A_eig)/min(A_eig)
            (B_eig, B_evec) = np.linalg.eig(BS)
            Bcond = max(B_eig)/min(B_eig)
            (C_eig, C_evec) = np.linalg.eig(CS)
            Ccond = max(C_eig)/min(C_eig)
             
            if (min(A_eig)< 1e-10 or min(B_eig < 1e-10) or min(C_eig < 1e-10) or Acond > 1e10 or Bcond > 1e10 or Ccond > 1e10):
                run=True
            else:
                run=False
                count = count+1
        
        # Verify Output
        print("Dim: ", d)
        print("Dist PSD:", np.sqrt(WBManifold.distance(AS, BS)), " ", np.sqrt(WBManifold.distance(BS, CS)))
        print("Dist mean:", np.sqrt(EucManifold.distance(Am, Bm)), " ", np.sqrt(EucManifold.distance(Bm, Cm)))
        print("A condition #: ", Acond, " B condition #: ", Bcond, " C condition #: ", Ccond)
        print("tries: ", count)
        print("")
        
        # Generate intermediate interpolated distributions
        tAll = np.linspace(0,1,nSamp)
        datM = np.zeros((2*nSamp, d,1))
        datS = np.zeros((2*nSamp, d,d))
        X = np.zeros((2*nSamp,3))

        count = 0
        n = d*20
        # Transition from 1->2
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
        X[:nSamp,0] = (1-tAll)
        X[:nSamp,1] = tAll

        # Transition from 2->3
        for t in tAll:
            tM = Bm * (1-t) + Cm * t
            tS = WBManifold.geodesic(BS, CS, t)
            run = True
            while (run):
                samps = np.random.multivariate_normal(np.squeeze(tM), tS, n)
                if (min(np.linalg.eig(np.cov(np.transpose(samps)))[0])>0):
                    run=False
            datM[count,:,:] = np.expand_dims(np.mean(samps,0),1)
            datS[count,:,:] = np.cov(np.transpose(samps))
            count = count + 1
        X[nSamp:,1] = (1-tAll)
        X[nSamp:,2] = tAll

        # Compute distance        
        d1=[]
        d2=[]
        d3=[]
        for i in range(2*nSamp):
            d1.append(WBManifold.distance(AS, datS[i]))
            d2.append(WBManifold.distance(BS, datS[i]))
            d2.append(WBManifold.distance(CS, datS[i]))

        mus = np.zeros((3,d))            
        mus[0,:] = np.squeeze(Am)
        mus[1,:] = np.squeeze(Bm)
        mus[2,:] = np.squeeze(Cm)
        covs = np.zeros((3,d,d))
        covs[0,:] = AS
        covs[1,:] = BS
        covs[2,:] = CS

        
        # Save the Data
        scio.savemat(folderOutput + '/dim_' + str(d) + '_'+str(nI)+ '.mat', mdict={'datM':datM, 'datS':datS, 'X':X, 'dim':d, 'd1':d1, 'd2':d2, 'mus':mus, 'covs':covs})
    