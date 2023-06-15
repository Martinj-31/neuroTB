

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:53:16 2022

@author: my
"""
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
'''
Find the rmse between the two weight lists.
groundTruth, target(list)

return y is float.   

'''
def getResults (groundTruthNetwork, groundTruthWeight, target, wmax):
    # # old version
    # groundTruthMax = max(np.array(groundTruthWeight).flatten().tolist())
    # groundTruth = copy.deepcopy(groundTruthNetwork)
    # for i in range(len(groundTruthNetwork)):
    #     del groundTruth[i][i]
    
    # groundTruth = sum(groundTruth, [])
    # groundTruthConnection = copy.deepcopy(groundTruth)
    # cnt = 0
    # for i in range(len(groundTruth)):
    #     if groundTruth[i] == 1:
    #         groundTruth[i] = groundTruthWeight[cnt][0]/groundTruthMax
    #         cnt += 1
    
    # for i in range (len(target)) :      
    #     target[i] = target[i] / wmax
    
    # TP=0
    # TN=0
    # FP=0
    # FN=0
    
    # for i in range(len(groundTruth)):
    #     if groundTruthConnection[i]==1 and target[i]>0.5:
    #         TP+=1
    #     elif groundTruthConnection[i]==1 and target[i]<=0.5:
    #         FN+=1
    #     elif groundTruthConnection[i]==0 and target[i]<=0.5:
    #         TN+=1
    #     elif groundTruthConnection[i]==0 and target[i]>0.5:
    #         FP+=1
    #     else:
    #         pass
    
    # new version
    groundTruthMax = max(np.array(groundTruthWeight).flatten().tolist())
    groundTruth = copy.deepcopy(groundTruthNetwork)
    for i in range(len(groundTruthNetwork)):
        del groundTruth[i][i]
    
    groundTruth = sum(groundTruth, [])
    groundTruthConnection = copy.deepcopy(groundTruth)
    cnt = 0
    for i in range(len(groundTruth)):
        if groundTruth[i] == 1:
            groundTruth[i] = groundTruthWeight[cnt][0]/groundTruthMax
            cnt += 1
    
    for i in range (len(target)) :      
        target[i] = target[i] / wmax

    TP=0
    TN=0
    FP=0
    FN=0
    
    for i in range(len(groundTruth)):
        if groundTruth[i]>=0.5 and target[i]>=0.5:
            TP+=1
        elif groundTruth[i]>=0.5 and target[i]<0.5:
            FN+=1
        elif groundTruth[i]<0.5 and target[i]<0.5:
            TN+=1
        elif groundTruth[i]<0.5 and target[i]>=0.5:
            FP+=1
        else:
            pass
        
    MCC = ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    
    # MCC = 0
    
    # RMSE = math.sqrt(sum((np.array(groundTruth)-np.array(target))**2)/len(groundTruth))
    
    return MCC, TP, TN, FP, FN


def ROC(groundTruthNetwork,weight) :
    groundTruth = copy.deepcopy(groundTruthNetwork)
    for i in range(len(groundTruthNetwork)):
        del groundTruth[i][i]
    
    groundTruth = sum(groundTruth, [])


    fpr, tpr, thresholds = roc_curve(groundTruth, weight)
    roc_auc = auc(fpr, tpr)
    
    print(roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")
    plt.savefig('roc',dpi=1600)
    plt.show()    
    
    


    