import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mostf(K):
    c = K[:,7]
    counts = 0
    countc = 0
    countq = 0
    temp = c
    for i in c:
        if str(i)=='S':
            counts = counts + 1
        elif str(i) == 'C':
            countc = countc + 1
        elif str(i) == 'Q':
            countq = countq + 1
    
    mostfreq = ''
    if max(counts,countc,countq)==counts:
        mostfreq = 'S'
    elif max(counts,countc,countq)==countc:
        mostfreq = 'C'
    elif max(counts,countc,countq)==countq:
        mostfreq = 'Q'
    
    count = 0
    for i in temp:
        
        if i!='S' and i!='C' and i!='Q':
            c[count] = mostfreq
            count = count + 1
        else:
            c[count] = str(i)
            count = count + 1
    K[:,7] = c
    return K

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, :8].values
y = dataset.iloc[:,8].values
testset = pd.read_csv('Testk.csv')
Xtest = testset.iloc[:, :8].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 3:4])
X[:, 3:4] = imputer.transform(X[:, 3:4])
imputer = imputer.fit(Xtest[:, 3:4])
Xtest[:, 3:4] = imputer.transform(Xtest[:, 3:4])
imputer = imputer.fit(Xtest[:, 6:7])
Xtest[:, 6:7] = imputer.transform(Xtest[:, 6:7])

X = mostf(X)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
Xtest[:, 2] = labelencoder_X.fit_transform(Xtest[:, 2])
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
Xtest[:, 7] = labelencoder_X.fit_transform(Xtest[:, 7])
'''
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
Xtest = onehotencoder.fit_transform(Xtest).toarray()
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
Xtest = sc.fit_transform(Xtest)


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 5,init = 'uniform',activation = 'relu',input_dim = 8) )
classifier.add(Dense(output_dim = 5,init = 'uniform',activation = 'relu') )
classifier.add(Dense(output_dim = 5,init = 'uniform',activation = 'relu') )
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'relu') ) #Activation Method would be Sigmoid here

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
classifier.fit(X,y,batch_size = 10,nb_epoch = 100)

y_pred = classifier.predict(Xtest)
y_pred = (y_pred>0.5)


'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 150,criterion = 'entropy',random_state = 0)
classifier.fit(X,y) 
# Predicting the Test set results
y_pred = classifier.predict(Xtest)
'''

'''
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',random_state = 0)
classifier.fit(X,y)

y_pred = classifier.predict(Xtest)
'''
result = []
for i in y_pred:
    if i == False:
        result.append(0)
    else :
        result.append(1)

df = pd.DataFrame(result, columns=["Survived"])
df.to_csv('NeuralNetwork.csv', index=False)
