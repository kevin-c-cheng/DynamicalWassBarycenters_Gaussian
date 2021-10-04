# -*- coding: utf-8 -*-

import ot
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib
import scipy.linalg
from MatrixSquareRoot import *

def PSD_Manifold_Exp(Sigma, tan):
    return np.matmul(Sigma, scipy.linalg.expm(np.matmul(np.linalg.inv(Sigma), tan)))

def PSD_Manifold_Exp2(Sigma, tan):
    SigmaSqrt = scipy.linalg.sqrtm(Sigma)
    SigmaSqrtInv = np.linalg.inv(Sigma)
    return np.matmul(np.matmul(SigmaSqrt, scipy.linalg.expm(np.matmul(np.matmul(SigmaSqrtInv, tan), SigmaSqrtInv))), SigmaSqrt)

def OptimalTransportDistance(x,y):
    cost = PNorm_CostMatrix(x,y)
    return OptimalTransportCostDistance.apply(cost)
    
def CovarianceBarycenter(covs, weights, S0, nIter=2):
        Sn_rt = []
        Sn_rt.append(MatrixSquareRoot.apply(S0))
        for i in range(nIter):
#            Sn_rt.append(MatrixSquareRoot.apply(torch.mm(torch.mm(torch.inverse(Sn_rt[i]), torch.einsum('a,abc->bc',weights, torch.matmul(torch.matmul(Sn_rt[i], covs), Sn_rt[i]))), torch.inverse(Sn_rt[i]))))
            Sn_rt.append(MatrixSquareRoot.apply(
                                    torch.mm(torch.mm(torch.inverse(Sn_rt[i]), 
                                    torch.matrix_power(torch.einsum('a,abc->bc',weights, 
                                    MatrixSquareRootT.apply(torch.matmul(torch.matmul(Sn_rt[i], covs), Sn_rt[i]))), 2)), 
                                    torch.inverse(Sn_rt[i]))))
        return torch.mm(Sn_rt[-1],Sn_rt[-1])

def CovarianceSymmetric_fromLD(L, diag):
    idx = np.tril_indices(len(diag), k=-1)
    idxR = (idx[1], idx[0])
    if (torch.is_tensor(L)):
        out = torch.diag(diag)
    else:
        out = np.diag(diag)
    out[idx] = L
    out[idxR] = L    
    return out

def CovarianceSymmetric_toLD(cov):
    idx = np.tril_indices(len(cov), k=-1)
    if (torch.is_tensor(cov)):
        return (cov[idx], torch.diag(cov))
    else:
        return (cov[idx], np.diag(cov))
    
def CovarianceSymmetricS_fromLDMu(L, diag, mu, s):
    idx = np.tril_indices(len(diag)+1, k=-1)
    idxR = (idx[1], idx[0])
    if (torch.is_tensor(L)):
        out = torch.diag(torch.cat((diag, s)))
        Lmu = torch.cat((L,mu))
    else:
        out = np.diag(np.concatenate((diag, s)))
        Lmu = np.concatenate((L, mu))
    out[idx] =  Lmu
    out[idxR] = Lmu  
    return out

def CovarianceSymmetricS_toLDMu(cov):
    #Returns: L, D, mu, s
    idx = np.tril_indices(len(cov), k=-1)
    if (torch.is_tensor(cov)):
        Lmu = cov[idx]
        DiagS = torch.diag(cov)
        return (Lmu[:-(len(cov)-1)], DiagS[:-1], Lmu[-(len(cov)-1):], DiagS[-1:])
    else:
        Lmu = cov[idx]
        DiagS = np.diag(cov)
        return (Lmu[:-(len(cov)-1)], DiagS[:-1], Lmu[-(len(cov)-1):], DiagS[-1:])
    
def CovarianceLLDecomposition(L, diag):
    idx = np.tril_indices(len(diag), k=-1)
    out = torch.diag(diag)
    out[idx] = L
    return out

def NormPdf(x,mu,cov):
    dim = len(mu)
    if len(x.shape)==1:
        return (2*np.pi)**(-dim/2) * 1/torch.sqrt(torch.det(cov))*torch.exp(-0.5*torch.matmul(torch.matmul((x-mu).float().T,torch.inverse(cov).float()),(x-mu).float()))
    else:
        tot = []
        for i in range(len(x)):
            tot.append((2*np.pi)**(-dim/2) * 1/torch.sqrt(torch.det(cov))*torch.exp(-0.5*torch.matmul(torch.matmul((x[i]-mu).float().T,torch.inverse(cov).float()),(x[i]-mu).float())))
        return torch.stack(tot)

def LogNormPdf(x,mu,cov):
    logdetS = torch.slogdet(cov)[1]
    xt = (torch.cat((x-mu, torch.ones(len(x),1)), dim=1)).t()

    log_q = -0.5 * (torch.sum(xt * torch.solve(xt, cov)[0], axis=1) + logdetS)
    return log_q

class OptimalTransportCostDistance(torch.autograd.Function):
# Pytorch autograd function to compute the optimal transport distance 
# given the cost matrix between the two measures
    @staticmethod
    def forward(ctx, cost):
        # Input is cost, output is sum(plan*cost)
        dim = cost.size()
        a=np.ones(dim[0])/dim[0]
        b=np.ones(dim[1])/dim[1]
        plan = torch.tensor(ot.emd(a,b, cost.detach().numpy())).float()
        ctx.plan = plan
        return torch.sum(cost*plan)

    @staticmethod
    def backward(ctx, grad_output):
        # gradient wrt cost is the plan
        plan = ctx.plan
        grad_input = grad_output.clone()
        return grad_input*plan

class OptimalTransportBarycenterStep(torch.autograd.Function):
# Pytorch autograd function for computing the Wasserstein Barycenter
# given the weights across the input measure (meas = list of point clouds)
# and prototype barycenter (bary).
    @staticmethod
    def forward(ctx, weights, meas, bary):
        step = 0.99
        n = len(weights)
        plans = []
        update = []
        for i in range(n):
            a=np.ones(len(bary))/len(bary)
            b=np.ones(len(meas[i]))/len(meas[i])

            cost = PNorm_CostMatrix(bary, meas[i])
            # maybe this can be a function
            plans.append(torch.tensor(ot.emd(a,b, cost.detach().numpy())).float())
            update.append(weights[i]*torch.matmul(plans[i]*len(bary),meas[i]))
        ctx.plans = plans
        ctx.weights = weights
        ctx.meas = meas
        return (1-step)*bary + step*torch.sum(torch.stack(update, dim=0),dim=0)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        step = 0.99
        plans = ctx.plans
        weights = ctx.weights
        meas = ctx.meas
        dim=(plans[0].size()[1], meas[0].size()[1])
        dWeights = torch.zeros(len(weights))
        for i in range(len(weights)):
            dWeights[i] = step*torch.sum(torch.matmul(plans[i],meas[i])*grad_input)
        
        dMeas = torch.zeros((len(weights),dim[0],dim[1]))
        for i in range(len(meas)):
            dMeas[i] = step*weights[i]*torch.matmul(plans[i],torch.ones(meas[i].size()))*grad_input
        
        dBary = (1-step)*grad_input
        return dWeights, dMeas, dBary

### We need to compute both the cost and the Barycenter for the eucliden pNorm and Wasserstein distance
# We start with the pnorm
def PNorm_Cost(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    c = torch.sum((torch.abs(x - y)) ** p, 2)
    return c


def PNorm_CostMatrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def PNormBarycenter(y, weights):
    # Frechet mean or centroid of points. y is n x d where n is the number of points, and d is the dimension
    # weights must sum to 1
    return torch.sum(torch.matmul(torch.diag(weights), y), dim=0)

def PNormBarycenter_Batch(y,weights):
    n = len(weights)
    baryAll = []
    for i in range(n):
        baryAll.append(PNormBarycenter(y, weights[i]))
    return baryAll


# This is for point clouds under the Wasserstein metric
def WassPointCloud_Plan(x,y, c, metric='sqeuclidean'):
    n1 = len(x)
    n2 = len(y)
    plan = ot.emd(np.ones(n1)/n1, np.ones(n2)/n2, c.detach().numpy())
    return torch.tensor(plan, dtype=torch.float)

def WassPointCloud_Cost(x,y,dtype=torch.float, metric='sqeuclidean'):
    c = PNorm_CostMatrix(x,y)
    plan = WassPointCloud_Plan(x,y,c, metric=metric)
    cost = torch.sum(plan*c)
    return cost

def WassPointCloud_CostMatrix(y, mu, dtype=torch.float):
    totalCost = []
    for t in range(len(y)):
        for k in range(len(mu)):
            totalCost.append(WassPointCloud_Cost(y[t], mu[k]))
    return torch.reshape(torch.stack(totalCost), (len(y), len(mu)))

def WassPointCloudBarycenter_Costs(bary, y):
    nMeas = len(y)
    plans=[]
    costs = []
    for i in range(nMeas):
        costs.append(PNorm_CostMatrix(bary, y[i]))
        plans.append(WassPointCloud_Plan(bary, y[i], costs[i]))
    return (costs, plans)


def WassPointCloudBarycenter_step(bary, y, weights, step = 0.8):
    # single step of the barycenter of the Wassestien point clouds as dictaed by Cuturi Doucet 2011 algorithm 2
    plans = WassPointCloudBarycenter_Costs(bary, y)[1]
    gradUp = []
    for i in range(len(plans)):
        gradUp.append(weights[i]*torch.matmul(plans[i]*len(y[0]), y[i]))
    
    baryUp = bary*(1-step) + step*torch.sum(torch.stack(gradUp), dim=0)
    return baryUp

def WassPointCloudBarycenter(bary, y, weights, step = 1e-2, maxIter=100, thresh = 1e-10):
    # Finds the barycenter of the Wassestien point clouds as dictaed by Cuturi Doucet 2011 algorithm 2
    iter = 0
    converge = False
    while (iter <= maxIter and converge == False):
        baryUp = WassPointCloudBarycenter_step(bary, y, weights, step=step)
        d = WassPointCloud_Cost(baryUp, bary)
        if (d<thresh):
            converge=True
        bary=baryUp
    return baryUp

def WassPointCloudBarycenter_BatchWeights_step(bary, y, weights, step = 1e-3, maxIter=100, thresh = 1e-3):
    # Finds the Wasserstein barycenters across various weights
    # Fixed y (means) barycenters and weights vary
    n = len(weights)
    baryAll = []
    for i in range(n):
        baryAll.append(WassPointCloudBarycenter_step(bary[i], y, weights[i]))
    return baryAll   
    
def WassPointCloudBarycenter_BatchWeights(bary, y, weights, step = 1e-3, maxIter=100, thresh = 1e-3):
    # Finds the Wasserstein barycenters across various weights
    # Fixed y (means) barycenters and weights vary
    n = len(weights)
    baryAll = []
    for i in range(n):
        baryAll.append(WassPointCloudBarycenter(bary[i], y, weights[i]))
    return baryAll   

def InitializeBary(nMeas, nSampMean, dim, mean=0, cov = None):
    if (cov.data is None):
        cov = torch.eye(dim)
    if (dim==1):
        dist = torch.distributions.normal.Normal(mean, cov)
    else:    
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
    baryAll =[]
    for i in range(nMeas):
        if (dim==1):
            baryAll.append(torch.randn((nSampMean,1)))
        else:    
            baryAll.append(dist.sample(torch.Size([nSampMean])))
    return baryAll
    
if __name__ == "__main__":    
    test = 'Barycenter_PSDManifoldS' #'CustomAutogradBary' #'testWassBarycenter_Means' #"testWassBarycenter"
    if (test == 'testDiffWass'):
        nSamp = 5
        dim = 2
        dtype = torch.float
        torch.manual_seed(0)
        x = torch.tensor(np.random.randn(nSamp, dim)*0.2, dtype=dtype, requires_grad=True)
        y = torch.tensor(np.random.randn(nSamp, dim)*0.2, dtype=dtype, requires_grad=False)+torch.ones(2)
        
        lr = 1e-1
        optimizer = torch.optim.SGD([x], lr = lr, momentum=0)
        nIter = 100
        for t in range(nIter):
            if (t==11):
                print('stop')
            cost=WassPointCloud_Cost(x,y)
            cost.backward()
            
            if (t % 1 == 0):
                print(str(t)+ ": cost: " +str(cost))
                plt.figure(1)
                plt.clf()
                plt.scatter(x[:,0].detach().numpy(), x[:,1].detach().numpy())
                plt.scatter(y[:,0].detach().numpy(), y[:,1].detach().numpy())
                plt.xlim([-2,2])
                plt.ylim([-2,2])
                plt.savefig("diffWass/image_{:03d}.png".format(t))
    
            x.data = x.data - lr*x.grad.data
            x.grad.data.zero_()
    #        optimizer.step()
    elif (test == 'testBatchDiffWass'):
        nSamp = 20
        nT = 100
        nDat = nT*3
        dtype = torch.float
        torch.manual_seed(0)
        dat = []
        for i in range(nT):
            dat.append(torch.tensor(np.random.randn(nSamp,2)*0.2+np.array([0,1]), dtype = dtype, requires_grad=False))
        for i in range(nT):
            dat.append(torch.tensor(np.random.randn(nSamp,2)*0.2+np.array([-1,-1]), dtype = dtype, requires_grad=False))
        for i in range(nT):
            dat.append(torch.tensor(np.random.randn(nSamp,2)*0.2+np.array([1,-1]), dtype = dtype, requires_grad=False))
        
        means = []
        nSampMeans = 200
        for i in range(3):
            means.append(torch.tensor(np.random.randn(nSampMeans,2), dtype=dtype, requires_grad=True) )
            
        nIter=100
        lr = 1e-1
        for t in range(nIter):
            cost = PNorm_CostMatrix(means, dat, dtype=dtype)
            cost2 = torch.sum(torch.min(cost, dim=1)[0])
            cost2.backward()
            
            for i in range(3):
                means[i].data = means[i].data - lr * means[i].grad.data
            
            if (t % 10 == 0):
                print(str(t)+ ": cost: " +str(cost2))
                plt.figure(1)
                plt.clf()
                tmp = torch.reshape(torch.stack(dat), (nSamp*nDat, 2))
                plt.scatter(tmp[:,0].detach().numpy(), tmp[:,1].detach().numpy())
                for k in range(3):
                    plt.scatter(means[k][:,0].detach().numpy(), means[k][:,1].detach().numpy())
                plt.xlim([-3, 3])
                plt.ylim([-3, 3])
                plt.savefig("diffWass/image_{:03d}.png".format(t))

            for i in range(3):
                means[i].grad.data.zero_()


        for i in range(3):
            print("mean: " + str(torch.mean(means[i], dim=0).detach().numpy()) + " std: " + str(torch.std(means[i], dim=0)))
    elif (test == 'testWassBarycenter'):
        nSamp = 20
        dtype = torch.float
        torch.manual_seed(0)
        dat = []
        dat.append(torch.tensor(np.random.randn(nSamp,2)*0.2+np.array([0,1]), dtype = dtype, requires_grad=False))
        dat.append(torch.tensor(np.random.randn(nSamp,2)*0.2+np.array([-1,-1]), dtype = dtype, requires_grad=False))
        dat.append(torch.tensor(np.random.randn(nSamp,2)*0.2+np.array([1,-1]), dtype = dtype, requires_grad=False))
        
        nSampMean = 20
        # Weights for barycentric mean
        pi = torch.tensor([0.4, 0.4, 0.2])
        bary = InitializeBary(1, nSampMean, 2, mean=torch.zeros(2), cov = torch.eye(2))
        out  = WassPointCloudBarycenter(bary[0], dat, pi)

        plt.figure(1)
        plt.clf()
        tmp = torch.reshape(torch.stack(dat), (nSamp*len(dat), 2))
        plt.scatter(tmp[:,0].detach().numpy(), tmp[:,1].detach().numpy())
        plt.scatter(out[:,0].detach().numpy(), out[:,1].detach().numpy())
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
#        plt.savefig("diffWass/image_{:03d}.png".format(t))
    elif (test == 'testWassBarycenter_Means'):
        nSamp = 20
        dim = 2
        dtype = torch.float
        torch.manual_seed(0)
        means = []
        means.append(torch.tensor(np.random.randn(nSamp,dim)*0.2+np.array([0,1]), dtype = dtype, requires_grad=True))
        means.append(torch.tensor(np.random.randn(nSamp,dim)*0.2+np.array([-1,-1]), dtype = dtype, requires_grad=True))
        means.append(torch.tensor(np.random.randn(nSamp,dim)*0.2+np.array([1,-1]), dtype = dtype, requires_grad=True))
        
        nSampMean = 20
        # Weights for barycentric mean
        pi = torch.tensor([0.2, 0.4, 0.4])
        gtBary = torch.tensor(np.random.rand(nSamp, dim), dtype=dtype, requires_grad=False)
        
        nIter = 10000
        learning_rate = 100
#        optimizer = torch.optim.Adam([means[0], means[1], means[2]], lr = learning_rate)
        for t in range(nIter):
            out  = WassPointCloudBarycenter_step(gtBary, means, pi)
            loss = WassPointCloud_Cost(out, gtBary)
            
            loss.backward()
            
            if (t % 100 ==0):
                print(loss)
                plt.figure(1)
                plt.clf()
                plt.scatter(means[0][:,0].detach().numpy(), means[0][:,1].detach().numpy())
                plt.scatter(means[1][:,0].detach().numpy(), means[1][:,1].detach().numpy())
                plt.scatter(means[2][:,0].detach().numpy(), means[2][:,1].detach().numpy())
                plt.scatter(out[:,0].detach().numpy(), out[:,1].detach().numpy())
                plt.xlim([-3, 3])
                plt.ylim([-3, 3])
                plt.savefig("images/image_{:03d}.png".format(t))


            for i in range(3):
                means[i].data = means[i].data - learning_rate*means[i].grad.data
            for i in range(3):
                means[i].grad.data.zero_()
            #optimizer.step()            
    elif (test == 'CustomAutogradDist'):
        # Verification of optimizing the Wasserstein distance with respect to an input measure
        x = torch.rand((4,2), requires_grad=True)
        y = torch.rand((4,2), requires_grad=True)
        for i in range(100):
            loss = OptimalTransportDistance.apply(x,y)
            # Loss is distance between measures
            loss.backward()
            # Here we decide for which point cloud we are optimizing for
            x.data = x.data - 1e-1 * x.grad.data
            if (i%10 ==0):                
                print(loss.detach().numpy())
                plt.figure(2)
                plt.clf()
                plt.scatter(x[:,0].detach().numpy(), x[:,1].detach().numpy())
                plt.scatter(y[:,0].detach().numpy(), y[:,1].detach().numpy())
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.savefig("images/image_{:03d}.png".format(i))

            x.grad.data.zero_()
            y.grad.data.zero_()
    elif (test == 'CustomAutogradBary'):
        # Verification of optimizing the Wasserstein barycenter with respect to the weights or the measure locations
        # Uses the gradient of the single step update from Cuturi Doucet 2014 Alg. 2
        # Assumes data is points in Rd, under square euclidean distance
        lr = 1
        dtype = torch.float
        dim = 2
        nSamp = 20
        muO = np.zeros((3,nSamp,dim))
        muO[0] = np.random.normal(0,1,(nSamp,dim))*0.2+[0,1]
        muO[1] = np.random.normal(0,1,(nSamp,dim))*0.2+[-1,-1]
        muO[2] = np.random.normal(0,1,(nSamp,dim))*0.2+[1,-1]
        mu = torch.tensor(muO, dtype = dtype, requires_grad=True)
        weights = torch.tensor([0.8, 0.1, 0.1], dtype = dtype, requires_grad=True)
        y = torch.tensor(np.random.normal(0,1,(nSamp,dim))*0.2 +[-1,-1], dtype = dtype)
        baryO = torch.tensor(np.random.normal(0,1,(nSamp,dim))*1.0, dtype = dtype, requires_grad=True)
        # Initialize the barycenter to be the converged estimate given the weights
        bary = torch.tensor(WassPointCloudBarycenter(baryO, mu, weights).detach().numpy(), dtype=dtype, requires_grad=True)
        for t in range(1000):
            baryUp = OptimalTransportBarycenterStep.apply(weights, mu, bary)
            # Loss is distance between barycenter and observation
            loss = torch.pow(OptimalTransportDistance(baryUp,y),2)
            loss.backward()
             
            if (t%100 ==0):                
                print(loss.detach().numpy())
                plt.figure(2)
                plt.clf()
                plt.scatter(mu[0][:,0].detach().numpy(), mu[0][:,1].detach().numpy(), c='b')
                plt.scatter(mu[1][:,0].detach().numpy(), mu[1][:,1].detach().numpy(), c='b')
                plt.scatter(mu[2][:,0].detach().numpy(), mu[2][:,1].detach().numpy(), c='b')
                plt.scatter(bary[:,0].detach().numpy(), bary[:,1].detach().numpy(), c='k')
                plt.scatter(y[:,0].detach().numpy(), y[:,1].detach().numpy(), c='r')
                plt.xlim([-2,2])
                plt.ylim([-2,2])
                plt.savefig("images/image_{:03d}.png".format(t))

            # HERE we swith between optimizing between weights or means
            weights.data = weights.data - lr * weights.grad.data
            weights.data = weights.clamp(min = 1e-8).data
            weights.data = weights.data/sum(weights.data)
#            mu.data = mu.data - lr * mu.grad.data
            bary.data = baryUp.data

            mu.grad.data.zero_()
            weights.grad.data.zero_()
#            bary.grad.data.zero_()
    elif (test == 'CovBary'):
        dtype = torch.float
        covsO = np.zeros((3,2,2))
        covsO[0] = [[1,0],[0,1]]
        covsO[1] = [[2,0],[0,2]]
        covsO[2] = [[3,0],[0,3]]
        S = torch.eye(2)
        weights = torch.tensor(np.ones(3)/3, dtype=dtype)
        
        covs = torch.tensor(covsO, dtype=dtype)
        covBary = CovarianceBarycenter(covs, weights, S, 1)
    elif (test == 'CovBarycenterMeans'):
        dtype = torch.float
        covsO = np.zeros((3,2,2))
        covsO[0] = [[1,0],[0,1]]
        covsO[1] = [[2,0],[0,2]]
        covsO[2] = [[3,0],[0,3]]
        S = torch.eye(2)
        weights = torch.tensor(np.ones(3)/3, dtype=dtype)
        
        covs = torch.tensor(covsO, dtype=dtype)
        covBary = CovarianceBarycenter(covs, weights, S, 1)
    elif (test == 'LLCov'):
        #Test that we can indeed do inference of a guassian using the LL decomposition of the covariance matrix
        dtype = torch.float
        dim = 2
        mean = [1,1]
        cov = [[5,2],[4,2]]
        y = torch.tensor(np.random.multivariate_normal(mean, cov, size=100), dtype=dtype, requires_grad=False)
        
        Lvec = torch.randn(int(dim*(dim-1)/2), dtype=dtype, requires_grad=True)
        diag = torch.randn(dim, dtype=dtype, requires_grad=True)
        mu = torch.randn(dim, dtype=dtype, requires_grad=True)
        
#        optimizer = torch.optim.Adam([Lvec, diag, mu], lr=1e-3)
        lr = 1e-3
        for i in range(10000):
            cov = CovarianceLLDecomposition(Lvec,diag)
            cost = -torch.sum(torch.log(NormPdf(y, mu, torch.matmul(cov,cov.T))))
            cost.backward()
            
#            optimizer.step()
            Lvec.data = Lvec.data - lr * Lvec.grad.data
            diag.data = diag.data - lr * diag.grad.data
            mu.data = mu.data - lr * mu.grad.data
            
            if (i % 100==0):
                print(cost)
        
            Lvec.grad.data.zero_()
            diag.grad.data.zero_()
            mu.grad.data.zero_()
    elif (test == 'Barycenter_LLCov'):
        torch.manual_seed(0)
        np.random.seed(0)
        costT=[]
        dtype = torch.float
        doPlot=True
        dim = 2
        meanA = np.array([0,0])
        meanB = np.array([20,1])
        covA = np.array([[5,0.2],[0.2,1]])
        covB = np.array([[1,0.2],[0.2,5]])
        meanAB = (meanA+meanB)/2
        covAB = CovarianceBarycenter(torch.tensor([covA,covB]).float(), torch.tensor([0.5, 0.5]).float(), torch.eye(2), nIter=10)
        
        nClust = 100
        x=torch.tensor(np.concatenate((np.tile([1,0], (nClust,1)), np.tile([0.5,0.5], (nClust,1)), np.tile([0,1], (nClust,1)))), dtype=dtype, requires_grad=False)
        y=torch.tensor(np.concatenate((np.random.multivariate_normal(meanA, covA, nClust), np.random.multivariate_normal(meanAB, covAB, nClust), np.random.multivariate_normal(meanB, covB, nClust))), dtype=dtype, requires_grad=False)
        
        uMeans=[]
        uL=[]
        uDiag=[]
        uMeans.append(torch.randn(dim, dtype=dtype, requires_grad=True))
        uMeans.append(torch.randn(dim, dtype=dtype, requires_grad=True))
#        uL.append(torch.randn(int(dim*(dim-1)/2), dtype=dtype, requires_grad=True))
#        uL.append(torch.randn(int(dim*(dim-1)/2), dtype=dtype, requires_grad=True))
#        uDiag.append(torch.randn(dim, dtype=dtype, requires_grad=True))
#        uDiag.append(torch.randn(dim, dtype=dtype, requires_grad=True))
        uL.append(torch.zeros(int(dim*(dim-1)/2), dtype=dtype, requires_grad=True))
        uL.append(torch.zeros(int(dim*(dim-1)/2), dtype=dtype, requires_grad=True))
        uDiag.append(torch.ones(dim, dtype=dtype, requires_grad=True))
        uDiag.append(torch.ones(dim, dtype=dtype, requires_grad=True))
        
        lr = 1e-3
        if (doPlot):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for t in range(10000):
            means = []
            covs = []
            out = []
            cov1 = CovarianceLLDecomposition(uL[0], uDiag[0])
            cov2 = CovarianceLLDecomposition(uL[1], uDiag[1])
            for i in range(nClust*3):
                means.append(uMeans[0]*x[i,0]+uMeans[1]*x[i,1])
                covs.append(CovarianceBarycenter(torch.cat((torch.matmul(cov1, cov1.T).unsqueeze(0),torch.matmul(cov2, cov2.T).unsqueeze(0)), dim=0), x[i], torch.eye(2)))
                out.append(NormPdf(y[i], means[i], covs[i]))
            
            cost = torch.sum(-torch.log(torch.stack(out).clamp(min=1e-6)))
            cost.backward()
            costT.append(cost.detach().numpy())
            
            uMeans[0].data = uMeans[0] - lr * uMeans[0].grad.data
            uMeans[1].data = uMeans[1] - lr * uMeans[1].grad.data
            uL[0].data = uL[0] - lr * uL[0].grad.data
            uL[1].data = uL[1] - lr * uL[1].grad.data
            uDiag[0].data = uDiag[0] - lr * uDiag[0].grad.data
            uDiag[1].data = uDiag[1] - lr * uDiag[1].grad.data
                
            if (t % 100==0):
                print(cost)
                print(torch.stack(uMeans).detach().numpy())
                print(torch.matmul(cov1, cov1.T).detach().numpy())
                print(torch.matmul(cov2, cov2.T).detach().numpy())
                if (doPlot):
                    ax.cla()
                    plt.ylim([-10, 10])
                    plt.xlim([-5, 25])
                    plt.scatter(y[:,0], y[:,1])
                    cov1 = CovarianceLLDecomposition(uL[0], uDiag[0])
                    cov2 = CovarianceLLDecomposition(uL[1], uDiag[1])
                    eig1 = np.linalg.eig(torch.matmul(cov1, cov1.T).detach().numpy())
                    eig2 = np.linalg.eig(torch.matmul(cov2, cov2.T).detach().numpy())
                    ell1 = pat.Ellipse(uMeans[0].data, eig1[0][0], eig1[0][1], np.arccos(eig1[1][0,0]), facecolor='none', edgecolor='red')
                    ell2 = pat.Ellipse(uMeans[1].data, eig2[0][0], eig2[0][1], np.arccos(eig2[1][0,0]), facecolor='none', edgecolor='red' )
                    ax.add_patch(ell1)
                    ax.add_patch(ell2)
                    
                    plt.savefig("LLCov/image_{:03d}.png".format(t))

            uMeans[0].grad.data.zero_()
            uMeans[1].grad.data.zero_()
            uL[0].grad.data.zero_()
            uL[1].grad.data.zero_()
            uDiag[0].grad.data.zero_()
            uDiag[1].grad.data.zero_()
    elif (test == 'Barycenter_PSDManifold'):
        torch.manual_seed(0)
        np.random.seed(0)
        dtype = torch.float
        doPlot=True
        dim = 2
        costT=[]

        meanA = np.array([0,0])
        meanB = np.array([20,1])
        covA = np.array([[5,0.2],[0.2,1]])
        covB = np.array([[1,0.2],[0.2,5]])
        meanAB = (meanA+meanB)/2
        covAB = CovarianceBarycenter(torch.tensor([covA,covB]).float(), torch.tensor([0.5, 0.5]).float(), torch.eye(2), nIter=10)
        
        nClust = 100
        x=torch.tensor(np.concatenate((np.tile([1,0], (nClust,1)), np.tile([0.5,0.5], (nClust,1)), np.tile([0,1], (nClust,1)))), dtype=dtype, requires_grad=False)
        y=torch.tensor(np.concatenate((np.random.multivariate_normal(meanA, covA, nClust), np.random.multivariate_normal(meanAB, covAB, nClust), np.random.multivariate_normal(meanB, covB, nClust))), dtype=dtype, requires_grad=False)
        
        uMeans=[]
        uL=[]
        uDiag=[]
        uMeans.append(torch.randn(dim, dtype=dtype, requires_grad=True))
        uMeans.append(torch.randn(dim, dtype=dtype, requires_grad=True))
        
        tmpL1 = np.random.randn(int(dim*(dim-1)/2))
        tmpD1 = np.random.randn(dim)+1
        (outL1,outD1)= CovarianceSymmetric_toLD(PSD_Manifold_Exp2(np.eye(2), CovarianceSymmetric_fromLD(tmpL1, tmpD1)))
        tmpL2 = np.random.randn(int(dim*(dim-1)/2))
        tmpD2 = np.random.randn(dim)+1
        (outL2,outD2)= CovarianceSymmetric_toLD(PSD_Manifold_Exp(np.eye(2), CovarianceSymmetric_fromLD(tmpL2, tmpD2)))
        uL.append(torch.tensor(outL1, dtype=dtype, requires_grad=True))
        uL.append(torch.tensor(outL2, dtype=dtype, requires_grad=True))
        uDiag.append(torch.tensor(outD1, dtype=dtype, requires_grad=True))
        uDiag.append(torch.tensor(outD2, dtype=dtype, requires_grad=True))
                
        lr = 1e-3
        if (doPlot):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for t in range(10000):
            means = []
            covs = []
            out = []
            cov1 = CovarianceSymmetric_fromLD(uL[0], uDiag[0])
            cov2 = CovarianceSymmetric_fromLD(uL[1], uDiag[1])
            for i in range(nClust*3):
                means.append(uMeans[0]*x[i,0]+uMeans[1]*x[i,1])
                covs.append(CovarianceBarycenter(torch.cat((cov1.unsqueeze(0), cov2.unsqueeze(0)), dim=0), x[i], torch.eye(2)))
                out.append(NormPdf(y[i], means[i], covs[i]))
            
            cost = torch.sum(-torch.log(torch.stack(out).clamp(min=1e-6)))
            cost.backward()

            costT.append(cost.detach().numpy())
           
            uMeans[0].data = uMeans[0] - lr * uMeans[0].grad.data
            uMeans[1].data = uMeans[1] - lr * uMeans[1].grad.data

            # Update covariance based on exponential map.            
            covTan1 = CovarianceSymmetric_fromLD(uL[0].grad.data, uDiag[0].grad.data)
            covExp1 = PSD_Manifold_Exp(cov1.detach().numpy(), -lr*covTan1.detach().numpy())
            (uL[0].data, uDiag[0].data) = CovarianceSymmetric_toLD(torch.tensor(covExp1))
            
            covTan2 = CovarianceSymmetric_fromLD(uL[1].grad.data, uDiag[1].grad.data)
            covExp2 = PSD_Manifold_Exp(cov2.detach().numpy(), -lr*covTan2.detach().numpy())
            (uL[1].data, uDiag[1].data) = CovarianceSymmetric_toLD(torch.tensor(covExp2))
                            
            if (t % 100==0):
                print("cost")
                print(cost)
                print("means")
                print(torch.stack(uMeans).detach().numpy())
                print("cov1")
                print(cov1.detach().numpy())
                print("cov2")
                print(cov2.detach().numpy())
                if (doPlot):
                    ax.cla()
                    plt.ylim([-10, 10])
                    plt.xlim([-5, 25])
                    plt.scatter(y[:,0], y[:,1])
                    cov1 = CovarianceSymmetric_fromLD(uL[0], uDiag[0])
                    cov2 = CovarianceSymmetric_fromLD(uL[1], uDiag[1])
                    eig1 = np.linalg.eig(cov1.detach().numpy())
                    eig2 = np.linalg.eig(cov2.detach().numpy())
                    ell1 = pat.Ellipse(uMeans[0].data, eig1[0][0], eig1[0][1], np.arccos(eig1[1][0,0]), facecolor='none', edgecolor='red')
                    ell2 = pat.Ellipse(uMeans[1].data, eig2[0][0], eig2[0][1], np.arccos(eig2[1][0,0]), facecolor='none', edgecolor='red' )
                    ax.add_patch(ell1)
                    ax.add_patch(ell2)
                    
                    plt.savefig("SymmCov/image_{:03d}.png".format(t))

            uMeans[0].grad.data.zero_()
            uMeans[1].grad.data.zero_()
            uL[0].grad.data.zero_()
            uL[1].grad.data.zero_()
            uDiag[0].grad.data.zero_()
            uDiag[1].grad.data.zero_()
    elif (test == 'Barycenter_PSDManifoldS'):
        torch.manual_seed(0)
        np.random.seed(0)
        costT=[]
        dtype = torch.float
        doPlot=True
        dim = 2
        meanA = np.array([0,0])
        meanB = np.array([20,1])
        covA = np.array([[5,0.2],[0.2,1]])
        covB = np.array([[1,0.2],[0.2,5]])
        meanAB = (meanA+meanB)/2
        covAB = CovarianceBarycenter(torch.tensor([covA,covB]).float(), torch.tensor([0.5, 0.5]).float(), torch.eye(2), nIter=10)
        
        nClust = 100
        x=torch.tensor(np.concatenate((np.tile([1,0], (nClust,1)), np.tile([0.5,0.5], (nClust,1)), np.tile([0,1], (nClust,1)))), dtype=dtype, requires_grad=False)
        y=torch.tensor(np.concatenate((np.random.multivariate_normal(meanA, covA, nClust), np.random.multivariate_normal(meanAB, covAB, nClust), np.random.multivariate_normal(meanB, covB, nClust))), dtype=dtype, requires_grad=False)
        
        uMeans=[]
        uL=[]
        uDiag=[]
        us = []

        tmpM1 = np.random.randn(dim)        
        S1=np.zeros((3,3))
        S1[:2,:2]=np.eye(2) + np.matmul(np.expand_dims(tmpM1,0).transpose(), np.expand_dims(tmpM1, 0))
        S1[-1,:2]=tmpM1
        S1[:2,-1]=tmpM1
        S1[-1,-1]=1
        (outL1,outD1, outM1, outS1)= CovarianceSymmetricS_toLDMu(S1)

        tmpM2 = np.random.randn(dim) + np.array([20,0])
        S2=np.zeros((3,3))
        S2[:2,:2]=np.eye(2) + np.matmul(np.expand_dims(tmpM2,0).transpose(), np.expand_dims(tmpM2, 0))
        S2[-1,:2]=tmpM2
        S2[:2,-1]=tmpM2
        S2[-1,-1]=1
        (outL2,outD2, outM2, outS2)= CovarianceSymmetricS_toLDMu(S2)

        uL.append(torch.tensor(outL1, dtype=dtype, requires_grad=True))
        uL.append(torch.tensor(outL2, dtype=dtype, requires_grad=True))

        uDiag.append(torch.tensor(outD1, dtype=dtype, requires_grad=True))
        uDiag.append(torch.tensor(outD2, dtype=dtype, requires_grad=True))
        
        uMeans.append(torch.tensor(outM1, dtype=dtype, requires_grad=True))
        uMeans.append(torch.tensor(outM2, dtype=dtype, requires_grad=True))

        us.append(torch.tensor(outS1, dtype=dtype, requires_grad=True))
        us.append(torch.tensor(outS2, dtype=dtype, requires_grad=True))
                
        lr = 1e-3
        if (doPlot):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for t in range(10000):
            means = []
            covs = []
            out = []
            cov1 = CovarianceSymmetric_fromLD(uL[0], uDiag[0])- 1/us[0]*torch.matmul(uMeans[0].unsqueeze(0).t(), uMeans[0].unsqueeze(0))
            cov2 = CovarianceSymmetric_fromLD(uL[1], uDiag[1])- 1/us[1]*torch.matmul(uMeans[1].unsqueeze(0).t(), uMeans[1].unsqueeze(0))
            for i in range(nClust*3):
                means.append(uMeans[0]*x[i,0]+uMeans[1]*x[i,1])
                covs.append(CovarianceBarycenter(torch.cat((cov1.unsqueeze(0), cov2.unsqueeze(0)), dim=0), x[i], torch.eye(2)))
                out.append(NormPdf(y[i], means[i], covs[i]))
            
            cost = torch.sum(-torch.log(torch.stack(out).clamp(min=1e-6)) )
            cost.backward()
            
            # Update covariance and means jointly based on exponential map of augmented matrix.            
            covTan1 = CovarianceSymmetricS_fromLDMu(uL[0].grad.data, uDiag[0].grad.data, uMeans[0].grad.data, torch.tensor([0]))
            covS1 = CovarianceSymmetricS_fromLDMu(uL[0].data, uDiag[0].data, uMeans[0].data, us[0].data)
            covExp1 = PSD_Manifold_Exp2(covS1.detach().numpy(), -lr*covTan1.detach().numpy())
            (uL[0].data, uDiag[0].data, uMeans[0].data, us[0].data) = CovarianceSymmetricS_toLDMu(torch.tensor(covExp1))
            
            covTan2 = CovarianceSymmetricS_fromLDMu(uL[1].grad.data, uDiag[1].grad.data, uMeans[1].grad.data, torch.tensor([0]))
            covS2 = CovarianceSymmetricS_fromLDMu(uL[1].data, uDiag[1].data, uMeans[1].data, us[1].data)
            covExp2 = PSD_Manifold_Exp2(covS2.detach().numpy(), -lr*covTan2.detach().numpy())
            (uL[1].data, uDiag[1].data, uMeans[1].data, us[1].data) = CovarianceSymmetricS_toLDMu(torch.tensor(covExp2))
            
            if (min(np.linalg.eig(covExp1)[0])<0 or min(np.linalg.eig(covExp2)[0])<0):
                print("stop)")
                            
            costT.append(cost.detach().numpy())
            if (t % 100==0):
                print("cost")
                print(cost)
                print("means")
                print(torch.stack(uMeans).detach().numpy())
                print("cov1")
                print(cov1.detach().numpy())
                print("cov2")
                print(cov2.detach().numpy())

                if (doPlot):
                    ax.cla()
                    plt.ylim([-10, 10])
                    plt.xlim([-5, 25])
                    plt.scatter(y[:,0], y[:,1])
                    cov1 = CovarianceSymmetric_fromLD(uL[0], uDiag[0])- 1/us[0]*torch.matmul(uMeans[0].unsqueeze(0).t(), uMeans[0].unsqueeze(0))
                    cov2 = CovarianceSymmetric_fromLD(uL[1], uDiag[1])- 1/us[1]*torch.matmul(uMeans[1].unsqueeze(0).t(), uMeans[1].unsqueeze(0))
                    eig1 = np.linalg.eig(cov1.detach().numpy())
                    eig2 = np.linalg.eig(cov2.detach().numpy())
                    ell1 = pat.Ellipse(uMeans[0].data, eig1[0][0], eig1[0][1], np.arccos(eig1[1][0,0]), facecolor='none', edgecolor='red')
                    ell2 = pat.Ellipse(uMeans[1].data, eig2[0][0], eig2[0][1], np.arccos(eig2[1][0,0]), facecolor='none', edgecolor='red' )
                    ax.add_patch(ell1)
                    ax.add_patch(ell2)
                    
                    plt.savefig("SymmCovS/image_{:03d}.png".format(t))

            uMeans[0].grad.data.zero_()
            uMeans[1].grad.data.zero_()
            uL[0].grad.data.zero_()
            uL[1].grad.data.zero_()
            uDiag[0].grad.data.zero_()
            uDiag[1].grad.data.zero_()
#            us[0].grad.data.zero_()
#            us[1].grad.data.zero_()
            
        
