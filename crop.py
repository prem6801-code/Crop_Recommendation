import numpy as np
import matplotlib as plt 
import pandas as pd 
import seaborn as sns


data = pd.read_csv('/content/Crop_recommendation (1).csv' )
print(data)

X=data.iloc[: , 0:7]
Y=data.iloc[: , -1]
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train  , Y_test = train_test_split(X ,Y , test_size=0.2 )

from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train , Y_train )

classifier_en =DecisionTreeClassifier(criterion='entropy')
classifier_en.fit(X_train , Y_train )

classifier.score(X_test , Y_test)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

sc.fit(X_train)

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

classifier_sc= DecisionTreeClassifier(criterion='gini')
classifier_sc.fit(X_train_sc , Y_train)
classifier_sc.score(X_test_sc , Y_test)

crop1=[50,49,55,30,46,6,150]

crop1=np.array([crop1])

classifier.predict(crop1)

pred=classifier.predict(crop1)

if pred[0] == 0 :
  print('rice')
elif pred[0] == 1:
  print('maize')
elif pred[0] == 2:
  print('chickpea')
elif pred[0] == 3:
  print('kidneybeans')
elif pred[0] == 4:
  print('pigeonpeas')
elif pred[0] == 5:
  print('mungbeans')
elif pred[0] == 6:
  print('blackgrams')
elif pred[0] == 7:
  print('lentils')
elif pred[0] == 8:
  print('pomegranate')
else:
  print('moyhbeans')

import pickle 
with open('model.pickle','wb') as f:
  pickle.dump(classifier,f)