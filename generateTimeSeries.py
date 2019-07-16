# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:33:41 2018

@author: Bijaya
"""

import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
import numpy as np


def computeRMSE(yg, yp):
    predictions = np.array(yp)
    targets = np.array(yg)
    rms = np.sqrt(((predictions - targets) ** 2).mean())
    return rms
    

def readData(filename):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    
    i = 1
    file = open(filename)
    line1= file.readline()
    
    vals = line1.strip().split()
    for val in vals:
        x1.append(i)
        i = i+1
        y1.append(float(val))
    
    line2 = file.readline()
    
    vals = line2.strip().split()
    for val in vals:
        x2.append(i)
        i = i+1
        y2.append(float(val))
        
    line3 = file.readline()
    
    vals = line3.strip().split()
    for val in vals:
        x2.append(i)
        i = i+1
        y2.append(float(val))
    return x1,y1,x2,y2

def readGroundtruth(filename):
    x = []
    y = []
    
    f = open(filename)
    f.readline()
    line = f.readline()
    
    vals = line.strip().split(':')
    for i in range(3, len(vals)):
        x.append(i-2)
        y.append(float(vals[i]))
    return x, y


def readEB(val):
    file = 'EBPRED201403.csv'
    x = []
    y = []
    
    f = open(file)
    #f.readline()
    line = f.readline()
    count = 1
    for line in f.readlines():
        vals = line.strip().split(',')
        x.append(count)
        y.append(float(vals[1]))
        count+=1
    return x[val-1:], y[val-1:]


year = 2013
val = 34


x1,y1,x2,y2 =  readData('predictedRNNSimple'+str(year)+str(val))   

#x3,y3,x4,y4 =  readData('predictedRegressionSimple'+str(20))

xg, yg = readGroundtruth('data/test')

print(len(yg))

xEB, yEB = readEB(val)

print(len(yEB))

print(len(yg), len(y1+y2))


#print(computeRMSE(yg[:-3], y3+y4))
 
fig = plt.figure()
plot = fig.add_subplot(111)

rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

plt.plot(xg, yg, 'b', label = "GroundTruth")
plt.plot(x1[0:52],y1[0:52],'r-', label = "Observed")  
plt.plot(x2[0:52],y2[0:52],'g--',label = "EpiDeep's Prediction")
plt.plot(xEB,yEB,'k.-',label = "Roni EB Prediction")

x = range(0,len(xg),5)
x_labels = [(i+20)%52+1 for i in x]


plt.xticks(x, x_labels)

for tick in plot.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)
    
for tick in plot.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)

#plt.title("National wILI Forecast (2016-2017)", fontsize=22, fontweight='bold')
plt.ylabel("weighted % ILI", fontsize=20, fontweight='bold')
plt.xlabel("Epidemiological Week", fontsize=20, fontweight='bold')


#plt.plot(x3[0:52],y3[0:52],'r+')  
#plt.plot(x4[0:52],y4[0:52],'y.',  label = "Regression's Prediction")  
plt.legend(loc = 2)
plt.ylim([0,6])
plt.savefig(str(year)+"Week"+str(val)+".png")
plt.show()