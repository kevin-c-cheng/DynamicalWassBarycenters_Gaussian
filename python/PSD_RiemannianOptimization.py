# Library for Riemannian manifold optimization using line search and gradient descent

import numpy as np
import scipy.linalg as spLin
import torch
from MatrixSquareRoot import *

def MatrixMultiply(mat):
    out = mat[0]
    for i in range(1,len(mat)):
        out = np.matmul(out, mat[i])
    return out

# Product Manifold
class Product():
    def __init__(self, manifolds):
        self.manifolds=manifolds
        self.n = len(manifolds)
        
    def distance(self, A,B):
        out = []
        for i in range(self.n):
            out.append(self.manifolds[i].distance(A[i], B[i]))
        return np.sum(out)

    def exp(self, A, tan):
        out = []
        for i in range(self.n):
            out.append(self.manifolds[i].exp(A[i], tan[i]))
        return out
    
    def riemannian(self, A,tan1, tan2):
        out = []
        for i in range(self.n):
            out.append(self.manifolds[i].riemannian(A[i], tan1[i], tan2[i]))
        return np.sum(out)
    
    def geodesic(self, A,B,t):
        out = []
        for i in range(self.n):
            out.append(self.manifolds[i].geodesic(A[i], B[i], t[i]))
        return out
    
    def euc_to_riemannian_gradient(self, X, tan):
        out = []
        for i in range(self.n):
            out.append(self.manifolds[i].euc_to_riemannian_gradient(X[i], tan[i]))
        return out

    def parallel_transport(self, A, B, tan):
        out = []
        for i in range(self.n):
            out.append(self.manifolds[i].parallel_transport(A[i], B[i], tan[i]))
        return out

# Euclidean manifold for matrices
class Euclidean():
    def distance(self, A, B):
        if (torch.is_tensor(A)):
            return torch.norm(A-B, p="fro")
        else:
            return np.linalg.norm(A-B)**2

    def dist(A, B):
        if (torch.is_tensor(A)):
            return torch.norm(A-B, p="fro")
        else:
            return np.linalg.norm(A-B)**2
    
    def exp(self, A, tan):
        return A + tan
    
    def riemannian(self, A,tan1, tan2):
        return np.inner(tan1.flatten(), tan2.flatten())
    
    def geodesic(self, A,B,t):
        return (1-t) *A + t*B
    
    def euc_to_riemannian_gradient(self, X, tan):
        return tan

    def parallel_transport(self, A, B, tan):
        return tan
    
# WB manifold
class WassersteinBuresPSDManifold():
    def distance(self, A, B):
        if (torch.is_tensor(A)):
            sqrtA = MatrixSquareRoot.apply(A)
            tmp = torch.matmul(torch.matmul(sqrtA, B), sqrtA)
            return torch.trace(A+B-2*MatrixSquareRoot.apply(tmp))
        else:
            dim = int(np.sqrt(A.size))
            A = np.reshape(A, (dim,dim))
            B = np.reshape(B, (dim,dim))
            sqrtA = spLin.sqrtm(A)
            tmp = MatrixMultiply([sqrtA, B, sqrtA])
            return np.trace(A+B-2*spLin.sqrtm(tmp))

    def dist(A, B):
        if (torch.is_tensor(A)):
            sqrtA = MatrixSquareRoot.apply(A)
            tmp = torch.matmul(torch.matmul(sqrtA, B), sqrtA)
            return torch.trace(A+B-2*MatrixSquareRoot.apply(tmp))
        else:
            dim = int(np.sqrt(A.size))
            A = np.reshape(A, (dim,dim))
            B = np.reshape(B, (dim,dim))
            sqrtA = spLin.sqrtm(A)
            tmp = MatrixMultiply([sqrtA, B, sqrtA])
            return np.trace(A+B-2*spLin.sqrtm(tmp))
    
    
    def exp(self, A, tan):
        X = spLin.solve_lyapunov(A, tan)
        tmp = A + tan + MatrixMultiply([X, A, X])
        return 0.5*(tmp+tmp.T) # Enforce Symmetric
    
    def riemannian(self, A,tan1, tan2):
        X1 = spLin.solve_lyapunov(A, tan1)
        X2 = spLin.solve_lyapunov(A, tan2)
        tmp = MatrixMultiply([X1, A, X2])
        return np.trace(0.5*(tmp+tmp.T)) # Enforce Symmetric
    
    def geodesic(self, A,B,t):
        sqrtA = spLin.sqrtm(A)
        invSqrtA= spLin.inv(sqrtA)
        tmp = (1-t)*A + t*spLin.sqrtm(MatrixMultiply([sqrtA, B, sqrtA]))
        return MatrixMultiply([invSqrtA, tmp, tmp, invSqrtA])
    
    def euc_to_riemannian_gradient(self, X, tan):
        tmp = MatrixMultiply([tan, X]) + MatrixMultiply([X, tan])
        return 0.5*(tmp+tmp.T) # Enforce symmetric

    def parallel_transport(self, A, B, tan):
        return tan

class LineSearchSimple(): #No step prediction
    def __init__(self, manifold, objective, contraction_factor = 0.5, suff_decrease= 0.5, maxIter = 25, init_alpha = 1.0, invalidCost = -1e99):
        self.manifold = manifold
        self.objective = objective
        self.contraction_factor = contraction_factor
        self.suff_decrease= suff_decrease # Paramter "c" in Algorithm 1 of "Optimization of Matrix Manifolds"
        self.maxIter = maxIter
        self.init_alpha = init_alpha
        self.invalid = invalidCost
        
    def checkPSD_tan(self, x0, tan, alpha):
        psdCheck=1
        if (isinstance(tan, list)): # Product Manifold
            newX = self.manifold.exp(x0, [x*alpha for x in tan])
            for check in newX:
                if (self.checkPSD(check)==0): # check if covariance is 
                    psdCheck=0
        else:
            newX = self.manifold.exp(x0, tan*alpha)
            if (self.checkPSD(newX)==0): # check if covariance is 
                psdCheck=0
        return (psdCheck, newX)

    def checkPSD(self, S):
        if (len(np.squeeze(S).shape) > 1 and np.min(np.linalg.eig(S)[0]) < 1e-5): # check if covariance is 
            return 0
        return 1
        
    def search(self, x0, tan):
        norm_d = np.sqrt(self.manifold.riemannian(x0, tan, tan))
        gradNorm = -norm_d**2

        f0 = self.objective.evaluate(x0)

        alpha = self.init_alpha
        if (isinstance(tan, list)): # Product Manifold
            newX = self.manifold.exp(x0, [x*alpha for x in tan])
        else:
            newX = self.manifold.exp(x0, tan*alpha)
            
        newF = self.objective.evaluate(newX)
        
        ite = 1
        psdCheck = 1
        while ( (newF > f0 + self.suff_decrease* alpha*gradNorm or psdCheck==0 or newF<self.invalid) and ite < self.maxIter):
            alpha *= self.contraction_factor
            (psdCheck, newX) =self.checkPSD_tan(x0, tan, alpha) # Chekc if new point is valid
            try:
                newF = self.objective.evaluate(newX)
            except:
                newF=0
                psdCheck=0
            ite +=  1

        if (newF > f0 or newF > f0 + self.suff_decrease* alpha*gradNorm or ite >= self.maxIter or newF<self.invalid):
            alpha = 0
            newX = x0
        print("cost:", newF)
        print("  step:", alpha)
        print("")        
        return newX
    
class GradientDescent():
    def __init__(self, manifold, step):
        self.manifold = manifold
        self.step = step
        
    def search(self, x0, tan):
        if (isinstance(tan, list)): # Product Manifold
            newX = self.manifold.exp(x0, [-x*self.step for x in tan])
        else:
            newX = self.manifold.exp(x0, -tan*self.step)
        return newX

class CostFunc():
    def __init__(self, A):
        self.A=A

    def evaluate(self, B):
        return WassersteinBuresPSDManifold.distance(self.A,B)
