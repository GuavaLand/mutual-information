import numpy as np
import matplotlib.pyplot as plt
import bsplineMI

def plotDistnCalcMI(dist1,dist2):
    plt.scatter(dist1,dist2,marker='.')
    plt.title('distribution 1 vs distribution 2')
    plt.axis('equal')
    plt.show()
    MI = bsplineMI.MIwrapper(dist1,dist2)
    print('The mutual information between dist1 and dist2: %f' %MI)
    return MI

# MI of two samples drawn from same distribution
sameDistMI = []
for i in range(100):
    dist1 = np.random.normal(0,2,1000)
    dist2 = np.random.normal(0,2,1000)
    sameDistMI.append(plotDistnCalcMI(dist1,dist2))


# MI of two samples drawn from same distribution, one shifted mean
sameDistMI_shiftMean = []
for i in range(100):
    dist1 = np.random.normal(0,2,1000)
    dist2 = np.random.normal(10,2,1000)
    sameDistMI_shiftMean.append(plotDistnCalcMI(dist1,dist2))

# MI of two samples drawn same mean, different std
sameDistMI_diffSTD = []
for i in range(100):
    dist1 = np.random.normal(0,2,1000)
    dist2 = np.random.normal(0,10,1000)
    sameDistMI_diffSTD.append(plotDistnCalcMI(dist1,dist2))

plt.hist(sameDistMI, alpha=0.5, label='same dist')
plt.hist(sameDistMI_shiftMean, alpha=0.5, label='shifted dist')
plt.hist(sameDistMI_diffSTD, alpha=0.5, label='different std')
plt.legend(loc='upper right')
plt.show()

# bimodal vs bimodal
bibi = []
for i in range(100):
    dist1 = np.concatenate([np.random.normal(0,2,500),np.random.normal(10,2,500)])
    dist2 = np.concatenate([np.random.normal(0,2,500),np.random.normal(10,2,500)])
    bibi.append(plotDistnCalcMI(dist1,dist2))

# bimodal vs unimodal
biuni = []
for i in range(100):
    dist1 = np.concatenate([np.random.normal(0,2,500),np.random.normal(10,2,500)])
    dist2 = np.random.normal(0,2,1000)
    biuni.append(plotDistnCalcMI(dist1,dist2))

plt.hist(bibi, alpha=0.5, label='bimodal bimodal')
plt.hist(biuni, alpha=0.5, label='bimodal unimodal')
plt.legend(loc='upper right')
plt.show() 

# from multi-variates normal
multi = []
for i in range(100):
    dist1, dist2 = np.random.multivariate_normal([0,0],[[1,20],[20,500]],1000).T
    multi.append(plotDistnCalcMI(dist1,dist2))
    
plt.hist(multi)
plt.show()