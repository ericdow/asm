from numpy import *
from cvxopt import matrix, solvers
import pylab

class Kriging:
    '''
    Inputs:
    asm    : instance of the ASM class
    m      : size of the original input space
    n      : size of reduced dimension
    Nd     : number of design points in each direction 
    '''
    def __init__(self, asm, m, n, Nd):

        self.asm = asm
        self.m  = m
        self.n  = n

        # construct the design points
        self.yd = self.design_points(asm.W,n,Nd)

        # build the kriging model
        self.C, self.fd = self.construct(self.yd,asm.W,n,\
                                                  asm.lamb)

    def __call__(self, x0):
        
        f0, e0 = self.evaluate(x0)

        return f0, e0

    def design_points(self,W,n,Nd):
        '''
        compute a set of design points
        Inputs:
        W   : singular vectors of the Jacobian covariance
        n   : dimension of the reduced domain
        Nd  : number of design points in each reduced dim

        Outputs:
        yd  : feasible design points
        '''

        m = W.shape[0]

        # bounding box for design points 
        W1 = W[:,:n]
        yl = -sum(W1*sign(W1),1)
        
        # construct hypercube of points between yl and -yl
        yh = zeros((n,Nd))
        for i in range(n):
            yh[i,:] = linspace(yl[i],-yl[i],Nd)

        # test if each point is feasible
        c = matrix(0., (m,1))
        A = matrix(W1.T)
        h = matrix(ones(2*m))
        G = vstack((eye(m),-eye(m)))
        G = matrix(G)
        dims = Nd*ones(n)
        yd = zeros(n)
        for i in range(Nd**n):
            ind = unravel_index(i,dims)
            yk = zeros(n)
            for j in range(n):
                yk[j] = yh[j,ind[j]]

            b = matrix(yk)
            sol = solvers.lp(c,G,h,A,b)

            # found a feasible point
            if sol['status'] == 'optimal':
                yd = vstack((yd,yk))

        # check if any feasible points found
        if len(yd.shape) == 1:
            print 'WARNING: no feasible design points found'
            yd = []
        else:
            yd = yd[1:,:]

        return yd
        

    def corr(self,y1,y2,lamb,n):

        sig2 = 2.*sqrt(self.m)/pi*sum(lamb)

        return exp(-sum(lamb[:n]*(y1-y2)**2)/2./sig2)
    
    def construct(self,yd,W,n,lamb):
        '''
        compute Cholesky factorization of covariance
        Inputs:
        yd   : design sites
        W    : singular vectors of the Jacobian covariance
        n    : dimension of the reduced domain
        lamb : eigenvalues of the Jacobian covariance

        Outputs:
        C    : Cholesky factor of the covariance matrix
        fd   
        '''

        # build the covaraince matrix
        sig2 = 2.*sqrt(self.m)/pi*sum(lamb)
        eta2 = 2.*sqrt(self.m)/pi*sum(lamb[n:])
        N = yd.shape[0]
        K = sig2*array([[self.corr(yd[i,:],yd[j,:],lamb,n)\
             for j in range(N)] for i in range(N)]) + eta2*eye(N)
        
        # ensure that K is positive definite
        if eta2 < (10 + N)*finfo(float).eps:
            K += (10 + N)*finfo(float).eps*eye(N)

        # compute Cholesky factor
        C = linalg.cholesky(K)

        # evaluate the model at the design sights
        fd = zeros(N)
        for i in range(N):
            fd[i] = self.asm(self.asm.transform(yd[i,:]))

        return C, fd

    def evaluate(self,x0):
        '''
        use the kriging model to interpolate the data
        Inputs:
        x0 : where to evaluate kriging surface

        Outputs:
        f0 : kriging interpolated value of f(x0)
        e0 : kriging error at x0
        '''

        lamb = self.asm.lamb
        W    = self.asm.W

        sig2 = 2.*sqrt(self.m)/pi*sum(lamb)
        eta2 = 2.*sqrt(self.m)/pi*sum(lamb[self.n:])

        # solve system to get weights
        N = self.yd.shape[0]
        y0 = dot(W[:,:self.n].T,x0)
        c0 = sig2*array([self.corr(y0,self.yd[i,:],lamb,self.n)\
                                         for i in range(N)])
        v = linalg.solve(self.C,c0)
        w = linalg.solve(self.C.T,v)

        # evaluate the kriging model
        f0 = dot(w,self.fd)

        # compute the kriging error
        e0 = sig2 + eta2 - dot(c0,w)
        
        return f0, e0
