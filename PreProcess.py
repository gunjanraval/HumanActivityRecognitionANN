# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:52:42 2018

@author: gunja
"""
import pandas as pd
col = []
for i in range(562):
    col.append(i)

df = pd.DataFrame(columns=col)

f1 = open('Y_test.txt')
activity = []
count=0
for i in f1:
    activity.append(i[0])
    count+=1
    

f = open('X_test.txt')
idx = 0
for i in f:
    col = []
    for j in i.split(' '):
        if(j!=''):
            col.append(j)
    
    col.append(activity[idx])
    df.loc[idx] = col
    idx+=1
    print(idx)
    
df.to_csv("TestDataSet.csv")


