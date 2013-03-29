from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from numpy import *
import pylab
from asm import ASM
from kriging import Kriging

def f(x):
    '''
    take in m-vector, return function value and gradient
    Inputs:
    x     : m-vector from input space

    Outputs:
    f     : function evaluated at x
    gradf : gradient of f evaluated at x
    '''
    m = len(x)
    random.seed(1)
    A = random.uniform(-1.,1.,(2,m))
    z = dot(A,x)

    f = z[0]*sin(z[1])
    gradzf = ((sin(z[1]),z[0]*cos(z[1])))

    gradf = dot(gradzf,A)

    return f, gradf

# number of Monte Carlo samples to estimate Jacobian covariance
Nsamples = 1000
# input space dimension
m = 10

# construct dimension reduction object 
asm = ASM(f,m,Nsamples)
W = asm.W
lamb = asm.lamb

# plot the eigenvalues
plt.semilogy(range(len(lamb)),lamb,'*')
plt.grid(True)
plt.xlabel('Index')
plt.ylabel('Eigenvalue')

# choose reduced dimension based on eigenvalue decay
n = 2
# number of design sites per reduced dimension 
Nd = 6
# construct the kriging surface
kr = Kriging(asm, m, n, Nd)
yd = kr.yd
fd = kr.fd

# evaluate surrogate at some points
N = 15
ys = zeros((N,N,n))
fs = zeros((N,N))
fe = zeros((N,N))
es = zeros((N,N))
y1 = linspace(min(yd[:,0]),max(yd[:,0]),N)
y2 = linspace(min(yd[:,1]),max(yd[:,1]),N)
for i in range(N):
    for j in range(N):
        ys[i,j,0] = y1[i]
        ys[i,j,1] = y2[j]
        xs = dot(W[:,:n],ys[i,j,:])
        fs[i,j], es[i,j] = kr(xs)
        fe[i,j], bar = f(xs)

# plot response surface
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(ys[:,:,0],ys[:,:,1],fs,rstride=1,\
        cstride=1,cmap=cm.jet,linewidth=0)
fig.colorbar(surf)
ax.scatter(yd[:,0],yd[:,1],fd,c='k')
plt.title('Kriging Surface')
 
# plot kriging error
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(ys[:,:,0],ys[:,:,1],es,rstride=1,\
        cstride=1,cmap=cm.jet,linewidth=0)
fig.colorbar(surf)
plt.title('Kriging Error')

# plot actual error
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(ys[:,:,0],ys[:,:,1],fe-fs,rstride=1,\
        cstride=1,cmap=cm.jet,linewidth=0)
fig.colorbar(surf)
plt.title('Actual Error')

plt.show()
