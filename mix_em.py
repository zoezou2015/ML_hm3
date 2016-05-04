from __future__ import division
import numpy as np
from numpy import *
from random import shuffle
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
from matplotlib import patches   
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt

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
#plot data points
#for i in xrange(len(x)):  
#	plt.plot(x[i][0], x[i][1], 'or')  
#plt.show()  


global miu
global p
global sigma


def initial_para(x, N, d):
	global miu
	global p
	global sigma
	
	#mean
	miu = np.zeros((2,2))
	miu[0] =  np.random.rand(1,2)	
	miu[1] =  -np.random.rand(1,2)
	
	#variance
	total_sigma = np.std(x)	
	sigma = np.array([total_sigma, total_sigma])

	#probability of each cluster
	p = np.zeros(2)
	p[0] = 1/2.0
	p[1] = 1 - p[0]

	return p, miu, sigma
	
def cal_var(x, miu, n, d, labels):
	temp = 0
	for i in range(len(x)):	
		temp += labels[i] * np.inner(np.add(x[i] ,-miu), np.add(x[i] ,-miu))
	#print n, d, temp / (n * d)
	return np.sqrt(temp / (n * d))

def Gaussian(x):
	w1 = p[0] / (2 * np.pi * sigma[0] ** 2)
	w2 = p[1] / (2 * np.pi * sigma[1] ** 2)
	w = np.array([w1, w2])
	t1 =  -np.inner(x - miu[0], x - miu[0]) / (2 * sigma[0] ** 2)
	t2 =  -np.inner(x - miu[1], x - miu[1]) / (2 * sigma[1] ** 2)
	t = np.array([t1, t2])
	return t , w

def log_likelihood(x, N):
	log_s = 0
	for i in range(N):
		[t, w] = Gaussian(x[i])
		log_s += logsumexp(t, b = w)
	return log_s

def e_step(x,N,d):
	global miu
	global p
	global sigma
	
	soft_p = np.zeros((N))
	labels = np.zeros(N)
	#e-step
	for i in range(N):
		[t, w] = Gaussian(x[i])
		#print 't',t
		#print 'w',w
		num1 = np.exp(logsumexp(t[0], b = w[0]))
		#print 'num',num1
		num2 = np.exp(logsumexp(t[1], b = w[1]))
		soft_p[i] = num1 / (num1 + num2)


		if soft_p[i] >= 0.5 :
			labels[i] = 1
		else:
			labels[i] = 0
	#print 'label',labels
	#print 'soft_p',soft_p	
	#print np.sum(soft_p)
	#return soft_p
	return labels

def m_step(x, N, d, labels):
	
	n = np.ones(d)
	n[0] = np.sum(labels)
	n[1] = N - n[0]
	#print 'n',n
	p[0] = n[0] / N
	p[1] = 1 - p[0]

	temp00 = 0
	temp01 = 0 
	temp10 = 0
	temp11 = 0
	for i in range(N):
		temp00 += labels[i] * x[i][0]
		temp01 += labels[i] * x[i][1]
		temp10 += (1-labels[i]) * x[i][0]
		temp11 += (1-labels[i]) * x[i][1]
	t1 = n[0]
	#if n[0] == 0:
	#	t1 = 1
	temp00 = temp00 / t1
	temp01 = temp01 / t1
	t2 = n[1]
	
	#if n[1] == 0:
	#	t2 = 1
	temp10 = temp10 / t2
	temp11 = temp11 / t2

	miu[0][0] = temp00 	
	miu[0][1] = temp01	
	miu[1][0] = temp10
	miu[1][1] = temp11

	sigma[0] = cal_var(x, miu[0], t1, d, labels)
	sigma[1] = cal_var(x, miu[1], t2, d, np.add(1, -labels))

def my_gmm(x,iter_num,Epsilon):  
	N = len(x)
	d = len(x[0])
	[p, miu, sigma] = initial_para(x, N, d)  

	print 'initial p = ', p 
	print 'initial miu = ', miu 
	print 'initial sigma = ', sigma
	log_function = log_likelihood(x, N)
	print 'log_function',log_function


	for i in range(iter_num):  
	
		labels = e_step(x, N, d) 
		#print labels 
		m_step(x, N, d, labels)  
		log_function = log_likelihood(x, N)
		#print 'log_function',log_function
	print '---------------------------------------'
	print 'loop p = ', i, 'fs',p 
	print 'loop miu = ', i,'fs', miu 
	print 'loop sigma = ', i, 'fs',sigma

	print 'log_function',log_function
	plot_data(x,labels)	
		#if sum(abs(Mu-Old_Mu)) < Epsilon:  
		#    break  

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

#if __name__ == '__main__':  
#   my_gmm(x, 100 , 0.001)  
	

