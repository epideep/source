import numpy as np

def createbatch(X,Y, Emd, num_batch):
    batchedData = []
    
    for i in range(0,len(X),int(len(X)/num_batch)):
        
        curX = X[i:i+int(len(X)/num_batch)]
        curY = Y[i:i+int(len(Y)/num_batch)]
        curEmd = Emd[ i:i+int(len(Y)/num_batch)]
        curBatch = (np.array(curX), np.array(curY), np.array(curEmd))
    batchedData.append(curBatch)
    return batchedData