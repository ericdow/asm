from numpy import *
import pylab

class ASM:
    '''
    Inputs:
    modfun   : function that takes in m-vector
    m        : input space dimension
    Nsamples : number of MC samples to estimate covariance
    '''
    def __init__(self, modfun, m, Nsamples):
        self.Ns = Nsamples
        self.m  = m
        self.mf = modfun

        # construct the samples
        samples = self.constructSamples()

        # compute the eigenvalues/vectors of Jacobian covariance
        self.lamb, self.W = self.eig_jac_cov(samples)

    def __call__(self,y):
        
        n = len(y)
        if n == self.m:
            x = copy(y)
        else:
            x = dot(self.W[:,:n], y)
        f, g = self.mf(x)

        return f

    def transform(self,xy):

        n = len(xy)
        if n == self.m:
            return dot(self.W[:,:n].T, xy)
        else if n < self.m:
            return dot(self.W[:,:n], xy)
        else:
            print 'Error: n > m'

    def constructSamples(self):
        '''
        Inputs:
        TODO density : 
        TODO domain  :

        Outputs:
        samples  : (Nsamples x m) array of samples
        '''

        samples = random.uniform(-1.0,1.0,(self.Ns,self.m))

        return samples

    def eig_jac_cov(self,samples):
        '''
        partitions the input space based on decay of the 
        eigenvalues of the Jacobian covariance

        Inputs:
        samples : where to sample model
        TODO where to truncate...

        Outputs:
        lamb    : eigenvalues of Jacobian covariance
        W       : eigenvectors " "
        '''
    
        # construct estimate of Jacobian covariance
        G = zeros((self.m,self.Ns))
        for i in range(self.Ns):
            foo, G[:,i] = self.mf(samples[i,:])
        G /= sqrt(self.Ns)

        # do an SVD
        W, S, V = linalg.svd(G)
        lamb = S*S

        return lamb, W

