from __future__ import division
import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
from matplotlib import patches   
from matplotlib.patches import Ellipse, Circle

import hard_em



def main():

	path = '/Users/Zoe/documents/PhD/Semester Jan-Apr 2015/Machine Learning/HW3/data.txt'
	x = []   
	fileIn = open(path)  
	for line in fileIn.readlines():  
		lineArr = line.strip().split(' ')  
		x.append([float(lineArr[0]),float(lineArr[1])])  
	x = np.array(x)
	#plot data
	plt.title('Initial data')
	for i in xrange(len(x)):  
		plt.plot(x[i][0], x[i][1], 'ob')  
	plt.show()  

	print "----------------hard EM--------------------"
	hard_em.my_gmm(x, 100, False, 0.0001)
	print "----------------soft EM--------------------"
	hard_em.my_gmm(x, 100, True, 0.0001)


if __name__ == '__main__':
    main()