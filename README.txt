asm
======

The python module 'dimred.py' can be used to perform dimensionality reduction. An object representing the reduced dimension input space can be created using

import asm

dr = dimred.Dimred(f,m,Nsamples)

where 'f' is a python function representing the model function, 'm' is the dimension of the input space (i.e. the dimension of the argument passed to 'f', and 'Nsamples' is the number of Monte Carlo samples used to estimate the Jacobian covariance matrix. Currently, the input sace is assumed to be uniformly distributed in the [-1,1]^m hypercube. The eigenvalues of the Jacobian covariance matrix are stored in 'dr.lamb'. The decay of these eigenvalues should guide the choice of the reduced dimension 'n'.

Once the dimension reduction object has been constructed and a size for the reduced input space has been chosen, a kriging surface can be constructed using 'kriging.py':

import kriging

kr = kriging(dr, m, n, Nd)

where 'dr' is the dimension reduction object, 'm' is the dimension of the full input space, 'n' is the dimension of the reduced input space, and 'Nd' is the number of design sites in each reduced dimension that will be used to construct the kriging surface. A total of Nd^n function evaluations will be made to construct the kriging surface. The cvxopt package (http://abel.ee.ucla.edu/cvxopt) is required to construct the kriging surface.

Once the kriging surface object has been created, the surface can be evaluated at the point x0 using

fk, ek = kr.evaluate(x0)

where 'fk' is the kriging estimate of the model function at x0 and 'ek' is the kriging error at x0.

See 'test_problem.py' for an example usage. 
