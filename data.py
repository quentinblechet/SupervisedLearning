import numpy as np
import pandas as pd

# get pandas dataframe
measureColList = [1,4,5,6,10,11,12,13,14,15,16,17,19,21,22,23,24,25,26,29,30,31,32,33,34]
quantifColList = [1,4,5,6,7,11,12,13,14,15]
dfM = pd.read_csv('Measure_300_40000_200.csv', sep=',', decimal='.', header=0, usecols=measureColList)
dfQ = pd.read_csv('Quantif_300_40000_200.csv', sep=',', decimal='.', header=0, usecols=quantifColList)
df = dfM.merge(dfQ, how='left')
df = df[df['Vol (pix)'] > 200]

df['Bounding_Square (pix)'] = (df['Xmax (pix)'] - df['Xmin (pix)']) * (df['Ymax (pix)'] - df['Ymin (pix)'])
df['Duration (pix)'] = df['Zmax (pix)'] - df['Zmin (pix)']
df['Mean Area'] = df['Vol (pix)'] / df['Duration (pix)']
df['Recovering'] = df['Mean Area'] / df['Bounding_Square (pix)']


df.to_csv('M&Q_300_40000_200.csv')
