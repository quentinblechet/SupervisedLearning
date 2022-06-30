from calendar import c
from email.base64mime import header_length
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay


df = pd.read_csv('M&Q_300_40000_200_label.csv', sep=',', decimal='.', header=0, index_col=0)
dfpca = pd.read_csv('PCA.csv', sep = ',', decimal = '.')


df = df[df['Vol (pix)'] > 200]
df = df[df['Label'] > 0]
# df = df[df['Label'] < 4]

df.loc[df['Label'] == 6, 'Label'] = 5

# Test
df.loc[df['Label'] == 2, 'Label'] = 1
df.loc[df['Label'] == 3, 'Label'] = 1
df.loc[df['Label'] == 4, 'Label'] = 0
df.loc[df['Label'] == 5, 'Label'] = 0
df.loc[df['Label'] == 6, 'Label'] = 0

group = 2131

#### Set up data in good shape with X_train, X_test, Y_train and Y_test

headerList = [
       'VolBounding (pix)', 'RatioVolbox', 'Vol (pix)', 'Surf (pix)',
       'SurfCorr (pix)', 'Comp (pix)', 'Spher (pix)', 'CompCorr (pix)',
       'SpherCorr (pix)', 'CompDiscrete', 'Ell_MajRad', 'Ell_Elon',
       'Ell_Flatness', 'volEllipsoid (unit)', 'RatioVolEllipsoid', 'AtCenter',
       'IntDen', 'Min', 'Max', 'Mean','Sigma', 'Bounding_Square (pix)', 
       'Duration (pix)', 'Mean Area','Recovering']

dimList = ['Dim.1','Dim.2','Dim.3','Dim.4','Dim.5']

X = []
X2 = []

for line in df.index:
    L = []
    L2 = []
    for column in headerList:
        L.append(df[column][line])
    for column2 in dimList:
        L2.append(dfpca[column2][line])
    X.append(L)
    X2.append(L2)

Y = []

for line in df.index:
    Y.append(df['Label'][line])

# With all variables
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,train_size=0.7, random_state=1)

# With PCA reduction dimension
# X_train, X_test, Y_train, Y_test = train_test_split(X2,Y,test_size=0.3,train_size=0.7, random_state=1)

clf = RandomForestClassifier(1000)
clf = clf.fit(X_train,Y_train)

prediction = clf.predict(X_test)
proba = clf.predict_proba(X_test)
print(accuracy_score(Y_test, prediction))
print(confusion_matrix(Y_test, prediction))
prediction = prediction.tolist()


from sklearn.metrics import roc_curve, auc

npY_test = np.array(Y_test)

fpr, tpr, thresholds = roc_curve(Y_test, proba[:, 1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.007, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()