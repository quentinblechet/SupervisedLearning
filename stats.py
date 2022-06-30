import pandas as pd
import numpy as np


df = pd.read_csv('M&Q_300_40000_200_label.csv', sep=',', decimal='.', header=0)

statList =  ['Vol (pix)','Bounding_Square (pix)', 'Duration (pix)', 'Mean Area', 'Recovering']
value = []
stats = {}


for i in range(5):
    value.append({})
    stats[i+1] = {}


for i in range(5):
    for header in statList:
        value[i][header] = []
        stats[i+1][header] = {}
        for row in df.index:
            if df['Label'][row] == i+1:
                value[i][header].append(df[header][row])


for i in range(len(value)):
    for header in statList:
        stats[i+1][header]['mean'] = np.mean(value[i][header])
        stats[i+1][header]['median'] = np.median(value[i][header])
        stats[i+1][header]['std'] = np.std(value[i][header])
        stats[i+1][header]['var'] = np.var(value[i][header])


for key, value in stats.items():
    print(key)
    for key2, value2 in value.items():
        print(key2)
        print(value2)

dfStats1 = pd.DataFrame.from_dict(stats[1])
dfStats2 = pd.DataFrame.from_dict(stats[2])
dfStats3 = pd.DataFrame.from_dict(stats[3])
dfStats4 = pd.DataFrame.from_dict(stats[4])
dfStats5 = pd.DataFrame.from_dict(stats[5])

dfStats1.to_csv('stats1.csv')
dfStats2.to_csv('stats2.csv')
dfStats3.to_csv('stats3.csv')
dfStats4.to_csv('stats4.csv')
dfStats5.to_csv('stats5.csv')