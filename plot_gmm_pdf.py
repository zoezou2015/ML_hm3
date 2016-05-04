"""
=============================================
Density Estimation for a mixture of Gaussians
=============================================

Plot the density estimation of a mixture of two Gaussians. Data is
generated from two Gaussians with different centers and covariance
matrices.
"""



from __future__ import division
import numpy as np
from numpy import *
from random import shuffle
from scipy.misc import logsumexp
from sklearn import mixture


path = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW3/data.txt'
x1 = []  
#y= []  
fileIn = open(path)  
for line in fileIn.readlines():  
	lineArr = line.strip().split(' ')  
	#print type(lineArr[2])
	x1.append([float(lineArr[0]),float(lineArr[1])])  
	#y.append(float(lineArr[1]))
#print len(x)
x = np.array(x1)

global miu
global p
global sigma


def initial_para(x, N, d):
	global miu
	global p
	global sigma
	
	
	#mean
	miu = np.zeros((2,2))	
	miu[0] = np.random.rand(1,2)	
	miu[1] = np.random.rand(1,2)
	

	#variance
	
	total_sigma = np.std(x)	
	sigma = np.array([total_sigma, total_sigma])

	#probability of each cluster
	p = np.zeros(2)
	p[0] = 1/2.0
	p[1] = 1 - p[0]

	return p, miu, sigma
#plot data points
def plot_data(x,labels):
	for i in xrange(len(x)): 
		if labels[i] >= 0.5: 
			plt.plot(x[i][0], x[i][1], 'or')  
		else:
			plt.plot(x[i][0], x[i][1], 'ob') 

	plt.plot(miu[0][0], miu[0][1], 'g*') 	
	plt.plot(miu[1][0], miu[1][1], 'g^') 
	x = y = np.arange(-4, 4, 0.1)
	x, y = np.meshgrid(x,y)
	plt.contour(x, y, (x-miu[0][0])**2 + (y-miu[0][1])**2, [sigma[0]**2])  
	plt.contour(x, y, (x-miu[1][0])**2 + (y-miu[1][1])**2, [sigma[1]**2])    
	plt.axis('scaled')
	plt.show()  

if __name__ == '__main__':  
	g = mixture.GMM(n_components=2,covariance_type='spherical', verbose = 1)
	print g.fit(x)
	print g.weights_
	print g.means_
	print g.covars_
	print g.get_params()
	print np.sum(g.score(x))



