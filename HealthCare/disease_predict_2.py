from pyexpat import model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC

import pickle
df = pd.read_csv('static/disease.csv')

cols = df.columns
data = df[cols].values.flatten()
s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)
df = pd.DataFrame(s, columns=df.columns)

df = df.fillna(0)
df1 = pd.read_csv('static/Symptom-severity.csv')

vals = df.values
symptoms = df1['Symptom'].unique()
for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)

d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination',0)
df = d.replace('foul_smell_of urine',0)

data = df.iloc[:,1:].values
labels = df['Disease'].values
data
x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.70)

from sklearn.naive_bayes import BernoulliNB
model= BernoulliNB()
model_2= model.fit(x_train, y_train)
# save the model to disk
with open('disease_model_2', 'wb') as files:
    pickle.dump(model_2, files)