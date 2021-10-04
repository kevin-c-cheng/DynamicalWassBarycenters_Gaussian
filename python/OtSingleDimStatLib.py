# -*- coding: utf-8 -*-
import numpy as np
import random as rd
import scipy as sp
import ot
import matplotlib.pyplot as plt
import multiprocessing
import scipy

    
def F_GInv(F,G):
    F=F.flatten()
    G=G.flatten()
    n = len(F)
    m = len(G)
    Fsort = np.sort(F)
    Gsort = np.sort(G)
    outX = np.zeros(m)
    outY = np.zeros(m)
    for i in range(m):
        dist = np.argwhere(Fsort <= Gsort[i])
        outY[i] = len(dist)/n
        outX[i] = (i+1)/m # cdf jumps at points in Gm
    return (outX, outY)

def DistanceToUniform_dep(p, cdf):
    #This only computes distance, we need squared distance. 
    # we assume that [0,0], and [1,1] are not included
    p=np.append(p,1)
    cdf = np.append(cdf,1)
    prevX = 0
    prevY = 0
    total = 0
    overUnder = 0

    for i in range(len(p)):
        if (cdf[i] < p[i]): # we are under
            if overUnder == 1: # we were over
                total += (np.abs(prevX-prevY) + (p[i]-prevY))/2 * (p[i]-prevX) # trapezoid
            elif overUnder !=-1: # we stayed under
                total += (np.abs(prevX-prevY) + (cdf[i]-prevY))/2 *(p[i]-prevX)
            overUnder = 1
        elif (cdf[i] < p[i]): # we are over
            if overUnder == -1: # and now we are under
                total += (np.abs(prevX-prevY) + 0)/2 * (prevY - prevX)
                total += (0 + (p[i]-cdf[i]))/2 * (p[i] - prevY)
            elif overUnder !=-1: # we are still over
                # we need to check if we fell under for some part
                if (p[i] < prevY): # if we did we have to integrate 2 smaller triangles
                    total += (np.abs(prevX-prevY))/2 * (prevY-prevX)
                    total += (p[i]-prevY)/2 * (p[i]-prevY)
                else:
                    total += (np.abs(prevY-prevX)+(cdf[i] - p[i]))/2 * (p[i]-prevX)
            overUnder = 0
        else:
            total+= np.abs(prevY-prevX)/2*(p[i]-prevX)
        
        prevX=p[i]
        prevY = cdf[i]
    return total

def DistanceSquaredToUniform(pp,cdf,step= 0.01):
    pp=np.append(pp,1)
    cdf = np.append(cdf,1)
    xAll = np.linspace(0,1,int(1/step)+1)
    total = 0 
    for x in xAll:
        argX = np.argwhere(pp>=x)
        total += (x - cdf[argX[0]])*(x - cdf[argX[0]])*step
    return total

def DistanceToUniform(pp,cdf,step= 0.01):
    pp=np.append(pp,1)
    cdf = np.append(cdf,1)
    xAll = np.linspace(0,1,int(1/step)+1)
    total = 0 
    for x in xAll:
        argX = np.argwhere(pp>=x)
        total += (x - cdf[argX[0]])*step
    return total

def MixtureDistanceToUniform(pp,cdfA,cdfB,step= 0.01):
    pp=np.append(pp,1)
    cdfA = np.append(cdfA,1)
    cdfB = np.append(cdfB,1)
    xAll = np.linspace(0,1,int(1/step)+1)
    total = 0 
    for x in xAll:
        argX = np.argwhere(pp>=x)
        total += (cdfA[argX[0]]-x)*(cdfB[argX[0]]-x)*step
    return total


def TwoSampleWTest(sampA, sampB, step = None):
    lenA = len(sampA)
    lenB = len(sampB)
    (cdfX, cdfY) = F_GInv(sampA,sampB)
            
    if (step is None):
        distOut = DistanceSquaredToUniform(cdfX, cdfY)*(lenA*lenB)/(lenA+lenB)
    else:
        distOut = DistanceSquaredToUniform(cdfX, cdfY, step=step)*(lenA*lenB)/(lenA+lenB)
    return distOut

def Compute2SampleWStat(dat, window, stride, step=None):
    lenDat = len(dat)
    dim=len(dat[0])
    out = np.zeros((int(np.floor(lenDat/stride)), dim))
    count = 0
    for i in range(0,lenDat-stride,stride):
        for j in range(dim):
            if (i<window or i >= lenDat-window):
                out[count,j]=0
            else:        
                win1 = dat[i-window:i,j]
                win2 = dat[i:i+window,j]
                out[count,j] = TwoSampleWTest(win1, win2, step=step)
        count = count+1
    outSingle = np.mean(out,axis=1)
    return (outSingle, out)

def ComputeSliced2SampleWStat(dat, window, stride, nProj = None):
    lenDat = len(dat)
    dim=len(dat[0])
    outMean = np.zeros(int(np.floor(lenDat/stride)))
    outMax = np.zeros(int(np.floor(lenDat/stride)))
    if (nProj is None):
        if (dim==1):
            nProj = 1
        elif (dim==2):
            nProj = 100
        elif (dim>=3):
            nProj=1000
    count = 0
    for i in range(0,lenDat-stride,stride):
        if (i<window or i >= lenDat-window):
            outMean[count]=0
            outMax[count]=0
        else:        
            win1 = dat[i-window:i]
            win2 = dat[i:i+window]
            (outMean[count], outMax[count]) = SlicedTwoSampleWstat(win1, win2, dim, nProj)
#        print('done with ' + str(i) + ' out of ' + str((len(dat)-stride)/stride))
        count = count+1
#    outSingle = np.sum(out,axis=1)
    return outMean

def RandomSampleStackPhased(dat, stack, nSamp):
    (lDat, dim) = dat.shape
    assert nSamp <= int(lDat / (2*stack-1))
    out = np.zeros((nSamp, stack*dim))
    use = np.ones(lDat-(stack-1))
    
    for i in range(nSamp):
        usable = np.argwhere(use==1).flatten()
        ind = rd.sample(list(usable), 1)[0]
        use[ind:ind+stack]=0
        out[i,:]=dat[ind:ind+stack,:].flatten()
    
    return out

def ComputeSliced2SampWStat_RandSamp_sub(i, window, lenDat, dat, stack, nSamp, nIter, dim, nProj):
    out = 0
    if (i<window or i >= lenDat-window):
        out = 0
    else:
        tmp=0
        for j in range(nIter):
            win1 = RandomSampleStackPhased(dat[i-window:i], stack, nSamp)
            win2 = RandomSampleStackPhased(dat[i:i+window], stack, nSamp)
            (out1, out2) = SlicedTwoSampleWstat(win1, win2, dim, nProj)
            tmp += out1
        out = tmp/nIter
    return out


def ComputeSliced2SampWStat_RandSamp(dat, window, stride, stack, nIter, nProj, multi=None):
    lenDat = len(dat)
    dim=len(dat[0])*stack
    outMean = np.zeros(int(np.floor(lenDat/stride)))
    outMax = np.zeros(int(np.floor(lenDat/stride)))

    nSamp = int(window/(2*stack-1))
    count = 0
    if (multi is None):
        for i in range(0,lenDat-stride,stride):
            outMean[count]= ComputeSliced2SampWStat_RandSamp_sub(i, window, lenDat, dat, stack, nSamp, nIter, dim, nProj)
#            if (i<window or i >= lenDat-window):
#                outMean[count]=0
#                outMax[count]=0
#            else:
#                tmp = 0
#                for j in range(nIter):
#                    win1 = RandomSampleStackPhased(dat[i-window:i], stack, nSamp)
#                    win2 = RandomSampleStackPhased(dat[i:i+window], stack, nSamp)
#                    (out1, out2) = SlicedTwoSampleWstat(win1, win2, dim, nProj)
#                    tmp += out1
#                outMean[count] = tmp / nIter
#            print('done with ' + str(i) + ' out of ' + str((len(dat)-stride)/stride))
            count = count+1
    else:
        pool = multiprocessing.Pool(multi)
        multiRes = [pool.apply_async(ComputeSliced2SampWStat_RandSamp_sub, (i,window,lenDat, dat, stack, nSamp, nIter, dim, nProj)) for i in range(0,lenDat-stride,stride)] 
        for res in multiRes:
            t = res.get(timeout=6000)
            outMean[count] = t
            count = count+1
#    outSingle = np.sum(out,axis=1)
    return outMean


def SlicedTwoSampleWstat(win1, win2, nDim, nProjections):
    out = np.zeros(nProjections)
    for i in range(nProjections):
        e = SlicedWassersteinProjectionDirection(nDim)
        win1_p=np.matmul(win1, e.transpose())
        win2_p=np.matmul(win2, e.transpose())
        out[i] = TwoSampleWTest(win1_p, win2_p)
    return (np.mean(out), np.max(out))

def ComputeSlicedOtDistance(dat, window, stride, nProj = None):
    lenDat = len(dat)
    dim=len(dat[0])
    out = np.zeros(int(np.floor(lenDat/stride)))
    if (nProj is None):
        if (dim==1):
            nProj = 1
        elif (dim==2):
            nProj = 100
        elif (dim>=3):
            nProj=1000
    count = 0
    for i in range(0,lenDat-stride,stride):
        if (i<window or i >= lenDat-window):
            out[count]=0
        else:        
            win1 = dat[i-window:i]
            win2 = dat[i:i+window]
            out[count] = SlicedOtDistance(win1, win2, dim, nProj)
        print('done with ' + str(i) + ' out of ' + str((len(dat)-stride)/stride))
        count = count+1
    return out

def ComputeSlicedOtDistance_RandSamp_sub(i, window, lenDat, dat, stack, nSamp, nIter, dim, nProj):
    out = 0
    if (i<window or i >= lenDat-window):
        out = 0
    else:
        for j in range(nIter):
            win1 = RandomSampleStackPhased(dat[i-window:i], stack, nSamp)
            win2 = RandomSampleStackPhased(dat[i:i+window], stack, nSamp)
            out += SlicedOtDistance(win1, win2, dim, nProj)
        out = out/nIter
    return out

def ComputeSlicedOtDistance_RandSamp(dat, window, stride, stack, nIter, nProj, multi=None):
    lenDat = len(dat)
    dim=len(dat[0])*stack
    out = np.zeros(int(np.floor(lenDat/stride)))

    nSamp = int(window/(2*stack-1))
    count = 0
    if (multi is None):
        for i in range(0,lenDat-stride,stride):
            out[count]=ComputeSlicedOtDistance_RandSamp_sub(i,window,lenDat, dat, stack, nSamp, nIter, dim, nProj)
#            if (i<window or i >= lenDat-window):
#                out[count]=0
#            else:
#                for j in range(nIter):
#                    win1 = RandomSampleStackPhased(dat[i-window:i], stack, nSamp)
#                    win2 = RandomSampleStackPhased(dat[i:i+window], stack, nSamp)
#                    out[count] += SlicedOtDistance(win1, win2, dim, nProj)
#                out[count] = out[count]/nIter
#            print('done with ' + str(i) + ' out of ' + str((len(dat)-stride)/stride))
            count = count+1
    else:
        pool = multiprocessing.Pool(multi)
#        multiRes = [pool.apply_async(ComputeSlicedOtDistance_RandSamp_sub, (i,window,lenDat, dat, stack, nSamp, nIter, dim, nProj)) for i in range(0,100,stride)] 
        multiRes = [pool.apply_async(ComputeSlicedOtDistance_RandSamp_sub,(i,window,lenDat, dat, stack, nSamp, nIter, dim, nProj)) for i in range(0,lenDat-stride,stride)] 
        for res in multiRes:
            t = res.get(timeout=6000)
            out[count] = t
            count = count+1

    return out

def SlicedOtDistance(win1, win2, nDim, nProjections):
    out = np.zeros(nProjections)
    for i in range(nProjections):
        e = SlicedWassersteinProjectionDirection(nDim)
        win1_p=np.matmul(win1, e.transpose())
        win2_p=np.matmul(win2, e.transpose())
        M = ot.utils.dist(win1_p, win2_p, metric = 'sqeuclidean')
        a= np.ones(len(win1_p))/len(win1_p)
        b = np.ones(len(win2_p))/len(win2_p)
        out[i] = np.sqrt(np.sum(np.multiply(ot.emd(a,b, M),M)))
#        out[i] = ot.emd_1d(win1_p, win2_p)
    return np.mean(out)

def ComputeOtDistance_RandSamp_sub(i, window, lenDat, dat, stack, nSamp, nIter):
    out = 0
    a = np.ones(nSamp)
    if (i<window or i >= lenDat-window):
        out = 0
    else:
        for j in range(nIter):
            win1 = RandomSampleStackPhased(dat[i-window:i], stack, nSamp)
            win2 = RandomSampleStackPhased(dat[i:i+window], stack, nSamp)
            M = ot.dist(win1, win2)    
            out += np.sum(np.multiply(ot.emd(a,a,M), M))
        out = out/nIter
    return out

def ComputeOtDistance_RandSamp(dat, window, stride, stack, nIter, nProj, multi = None):
    lenDat = len(dat)
    out = np.zeros(int(np.floor(lenDat/stride)))

    nSamp = int(window/(2*stack-1))
    count = 0
    if (multi is None):
        for i in range(0,lenDat-stride,stride):
            out[count]=ComputeOtDistance_RandSamp_sub(i,window,lenDat, dat, stack, nSamp, nIter)
            count = count+1
    else:
        pool = multiprocessing.Pool(multi)
        multiRes = [pool.apply_async(ComputeOtDistance_RandSamp_sub, (i,window,lenDat, dat, stack, nSamp, nIter)) for i in range(0,lenDat-stride,stride)] 
        for res in multiRes:
            t = res.get(timeout=6000)
            out[count] = t
            count = count+1

    return out

def SlicedWassersteinProjectionDirection(nDim):
    if (nDim==1):
        e=np.array([1])
    elif (nDim==2):
        x1 = rd.uniform(0,2*np.pi)
        e = np.array([np.cos(x1), np.sin(x1)])
    elif (nDim == 3):
        # uniform spherical sampling method from Marsaglia (1972). http://mathworld.wolfram.com/SpherePointPicking.html
        x1 = rd.uniform(-1,1)
        x2 = rd.uniform(-1,1)
        while (x1*x1 + x2*x2>=1):
            x1 = rd.uniform(-1,1)
            x2 = rd.uniform(-1,1)
        e = np.array([2*x1*np.sqrt(1-x1*x1-x2*x2), 2*x2*np.sqrt(1-x1*x1-x2*x2), 1-2*(x1*x1+x2*x2)])
    else:
        # Sample from multivariate normal and normalize
        x = np.random.multivariate_normal(np.zeros(nDim), np.eye(nDim))
        e=x/np.linalg.norm(x)
    return np.expand_dims(e,axis=0)

def SlidingWindowKS(dat, win):
    ks = np.zeros(len(dat))
    dim = len(dat[0])
    for i in range(len(dat)):
        if (i<=win or i>=len(dat)-win):
            ks[i]=0
        else:
            for d in range(dim):
                winA = dat[i-win:i,d]
                winB = dat[i:i+win,d]
                ks[i]+=scipy.stats.ks_2samp(winA, winB)[0]
    return ks/dim

def Compute1dOtDistance(cloud_i, cloud_j, metric='minkowski'):
    lenA = len(cloud_i)
    a = np.ones(lenA)/lenA
    out = 0
    if len(np.shape(cloud_i))==1:
        M = ot.dist(np.expand_dims(cloud_i,axis=1), np.expand_dims(cloud_j, axis=1), metric = metric)    
    else:
        M = ot.dist(cloud_i, cloud_j, metric = metric)    
    return np.sum(np.multiply(ot.emd(a,a,M), M))

def SlidingWindow1dOt(dat, win, metric='minkowski'):
    p1 = np.zeros(len(dat))
    dim = len(dat[0])
    for i in range(len(dat)):
        if (i<=win or i>=len(dat)-win):
            p1[i]=0
            d=1
        else:
            for d in range(dim):
                winA = dat[i-win:i,d]
                winB = dat[i:i+win,d]
                p1[i]+=Compute1dOtDistance(winA, winB, metric=metric)
    return p1/dim
        
#nSamp = 1000
#nRun = 100000
#distOut = np.zeros(nRun)
#for r in range(nRun):
#    x = np.random.normal(0, 1, nSamp)
#    y = np.random.normal(0, 1, nSamp)
#    
#    (cdfX, cdfY) = F_GInv(x,y)
#    distOut[r] = DistanceSquaredToUniform(cdfX, cdfY)*(nSamp*nSamp)/(2*nSamp)

#(histX, histY) = sp.histogram(distOut,)
#plt.hist(distOut, bins=100)
#len(np.argwhere(distOut>0.4612))/nRun
#for i in range(10):
#    print(np.sum(np.power(distOut,i+1))/nRun) #non centralized moments