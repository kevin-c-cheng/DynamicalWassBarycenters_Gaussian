# -*- coding: utf-8 -*-
import torch 
import numpy as np
import scipy.stats as scistats
import scipy.signal as scisig
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import DiffWassersteinLib as dwl
import PSD_RiemannianOptimization as psd
import OtSingleDimStatLib as otcpd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import ot

def GmmWassDistance(mus1, covs1, x1, mus2, covs2, x2):
    if (isinstance(mus1, list)):
        n1 = len(mus1)
    else: # We are working with Tensors
        n1 = mus1.shape[0]
    if (isinstance(mus2, list)):
        n2 = len(mus2)
    else: # We are working with Tensors
        n2 = mus2.shape[0]
        
    cost = torch.zeros((n1,n2))
    
    for i in range(n1):
        for j in range(n2):
            if (i!=j):
                cost[i,j] = (psd.WassersteinBuresPSDManifold.dist(covs1[i], covs2[j]) + psd.Euclidean.dist(mus1[i], mus2[j])**2)
    a1 = np.ones(n1)/n1
    a2 = np.ones(n2)/n2
    mapp = ot.emd(a1, a2, cost.detach().numpy())
    return torch.sum(torch.tensor(mapp, dtype = torch.float)*cost)

def GaussGmm_WassDist(mu, cov, mus2, covs2, x2):
    # We assume here that p1 is a single gaussian and p2 is multi
    if (isinstance(mus2, list)):
        n2 = len(mus2)
    else: # We are working with Tensors
        n2 = mus2.shape[0]
    cost = []    
    for j in range(n2):
        cost.append( x2[j]*(psd.WassersteinBuresPSDManifold.dist(cov, covs2[j]) + psd.Euclidean.dist(mu, mus2[j])**2))
    return torch.sqrt(torch.sum(torch.stack(cost)))

def GaussGmm_WassDist_MonteCarlo(mu, cov, mus2, covs2, x2, n=5000):
    # We assume here that p1 is a single gaussian and p2 is multi
    (K,dim)=np.shape(mus2.detach().numpy())
    p1 = scistats.multivariate_normal(mu, cov)
    s1 = p1.rvs(n)
    p2 = []
    for i in range(K):
        p2.append(scistats.multivariate_normal(mus2[i].detach().numpy(), covs2[i].detach().numpy()))
    p2I = scistats.multinomial(1,x2)
    s2 = np.zeros((n,dim))
    idx = p2I.rvs(n)
    for i in range(n):
        s2[i,:] = p2[np.squeeze(np.argwhere(idx[i]))].rvs(1)
        
    return ComputeOtDistance(s1, s2)


def GaussWassDistance(mu1, cov1, mu2, cov2):
    return (psd.WassersteinBuresPSDManifold.dist(cov1, cov2) + psd.Euclidean.dist(mu1, mu2)**2)

def GaussianSample(mu, sig):
    return scistats.multivariate_normal.rvs(mu, sig)

def LogGammaLiklihoodBimodalAB(p,a0,b0,a,b,w,delta):
    sp = torch.nn.Softplus(beta=10)
    mini = 1.1
    pDelta=[]
    for k in range(p.K):
        pDelta.append( torch.log(torch.max(
            w[k]*(torch.exp( (a0[k] -1)*torch.log(delta[:,k]) + (b0[k] -1)*torch.log(1-delta[:,k]) 
            + torch.lgamma(a0[k]+b0[k]) - torch.lgamma(a0[k]) - torch.lgamma(b0[k]) ))
            + (1-w[k])*(torch.exp( (a[k] -1)*torch.log(delta[:,k]) + (b[k] -1)*torch.log(1-delta[:,k]) 
            + torch.lgamma(a[k]+b[k]) - torch.lgamma(a[k]) - torch.lgamma(b[k]) ))
            , torch.tensor(1e-20))))
        
    log_pDelta = torch.stack(pDelta)
    return log_pDelta


def StateEvolutionDynamics(p, x0, pi, delta):
    gamma = delta*pi.repeat(p.T,1)
    Gamma = torch.sum(gamma,dim=1)

    X=[x0]
    for i in range(p.T):
        X.append(X[i]*(1-Gamma[i]) + gamma[i])
    X2=torch.stack(X[1:]).clamp(min=p.eps)
    return X2

def WindowData(datO, window, stride, offset=0):
    d = int(np.floor(window/2))
    if (offset < d): # offset has to be larger than d
        offset = d
        
    if (len(datO.shape) == 1):
        dat = np.expand_dims(datO,axis=1)
    
    out = []
#    for i in range(0, len(datO)-window, stride):
    for i in range(offset, len(datO)-offset, stride):
        if (window%2==0):
            out.append(datO[i-d:i+d,:])
        else:
            out.append(datO[i-d:i+(d+1),:])
        
    return np.asarray(out)


def LogGaussianDistribution(obs, mu = None, sig = None):
    if (hasattr(obs, "__len__")):
        if (mu is None):
            mu = 0
        if (sig is None):
            sig = 1
        return -sig-0.9189 - 0.5*torch.square((obs-mu)/sig)
    else:
        dim =len(obs)
        if (mu is None):
            mu = torch.zeros(dim)
        if (sig is None):
            sig = torch.eye(dim)
            
        if (np.isscalar(sig)):
            sig = sig*torch.eye(dim)
        # not debugged yet
        dim = len(obs)
        lik = -dim/2*torch.log(2*torch.tensor(np.pi)) -0.5*torch.logdet(sig) - 0.5*torch.matmul(torch.matmul(torch.unsqueeze(obs-mu,0), torch.inverse(sig)), torch.transpose(torch.unsqueeze(obs-mu,0),0,1))
        return lik.squeeze()
    
def PointCloud_WassersteinDistanceLiklihood(xCloud, clusterCloud, var):
    d = dwl.OptimalTransportDistance(xCloud,clusterCloud)
    return 1/(2*torch.tensor(np.pi)*torch.sqrt(var))*torch.exp(-0.5*torch.norm(d)/var)

def FitGMM_Model(dat,K):
    gmm = GaussianMixture(n_components=K).fit(dat)
    mu = []
    cov = []
    for i in range(K):
        mu.append(gmm.means_[i])
        cov.append(gmm.covariances_[i])
    return (mu,cov)

def ComputeOtDistance(w1, w2, metric = 'sqeuclidean'):
    m1 = np.ones(len(w1))/len(w1)
    m2 = np.ones(len(w2))/len(w2)
    
    M = ot.dist(w1,w2, metric=metric)
    S = ot.emd(m1,m2,M)
    return np.sum(np.multiply(M,S))

def CPD_WQT(dat,win):
    (L,dim) = dat.shape
    filt = np.zeros(2*win+1)
    for i in range(2*win+1):
        filt[i] = ((i-win)/win)**2
    cpd = np.zeros((dim,L))
    for d in range(dim):
        for i in range(win,L-win):
            cpd[d,i] = otcpd.TwoSampleWTest(dat[i-win:i], dat[i:i+win])
            
        cpd[d] = scisig.convolve(cpd[d], filt, 'same')
    return np.mean(cpd,0) 

def CPD_WM1(dat,win):
    (L,dim) = dat.shape
    filt = np.zeros(2*win+1)
    for i in range(2*win+1):
        filt[i] = np.abs((i-win)/win)
    cpd = np.zeros((dim,L))
    for d in range(dim):
        for i in range(win,L-win):
            cpd[d,i] = otcpd.Compute1dOtDistance(dat[i-win:i], dat[i:i+win])
            
        cpd[d] = scisig.convolve(cpd[d], filt, 'same')
    return np.mean(cpd,0) 

    
def CPD_Init(dat, cpd, thresh, K):
    (L,dim)=dat.shape
    pkIdx = scisig.find_peaks(cpd)[0]
    pkVal = cpd[pkIdx]
    n = sum(pkVal>thresh)
    if (n+1 < K):
        kPkIdx = np.linspace(0, len(cpd), K+1).astype(int)
        n=K-1
    else:
        kPkIdx = np.zeros(n+2).astype(int)
        kPkIdx[1:-1] = pkIdx[np.sort(np.argsort(pkVal)[-n:])]
        kPkIdx[-1]=len(cpd)

    aff = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            if (i!=j):
                aff[i,j] = ComputeOtDistance(dat[kPkIdx[i]:kPkIdx[i+1]], dat[kPkIdx[j]:kPkIdx[j+1]])
    clustering = SpectralClustering(n_clusters=K, affinity='precomputed').fit(np.exp(-aff))
    outM=[]
    outSig=[]
    for k in range(K):
        datT=np.zeros((0,dim))
        for i in range(n+1):
            if (clustering.labels_[i]==k):
                datT = np.append(datT,dat[kPkIdx[i]:kPkIdx[i+1]], axis=0)
                
        outM.append(np.mean(datT, 0))
        outSig.append(np.cov(datT.transpose()))

    return (outM, outSig)

def label_Init(dat, L):
    (n,dim)=dat.shape
    nL = np.unique(L)
    outM=[]
    outSig=[]
    for k in range(len(nL)):
        datT=np.squeeze(dat[np.argwhere(L==nL[k])])
                
        outM.append(np.mean(datT, 0))
        outSig.append(np.cov(datT.transpose()))

    return (outM, outSig)

def FitMuSig(dat,K):
    gmm = GaussianMixture(n_components=K).fit(dat)
    eig=[]
    for i in range(K):
        eig.append(np.linalg.eig(gmm.covariances_[i])[0])
    return (np.mean(gmm.means_,0), np.mean(eig)*np.eye(dat.shape[1]))

        