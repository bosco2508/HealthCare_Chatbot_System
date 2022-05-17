import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
df=pd.read_csv("static/Training_Disease.csv")
cols=df.columns
l1=list(cols[:-1])
disease=['Fungal infection','Allergy','Drug Reaction','Peptic ulcer diseae','Migraine','Paralysis (brain hemorrhage)','Jaundice','Chicken pox','Common Cold','Dimorphic hemmorhoids(piles)','Varicoseveins','Hypothyroidism','Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis','Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'Drug Reaction':2,
'Peptic ulcer diseae':3,'Migraine':4,'Paralysis (brain hemorrhage)':5,'Jaundice':6,'Chicken pox':7,'Common Cold':8,'Dimorphic hemmorhoids(piles)':9,
'Varicose veins':10,'Hypothyroidism':11,'Arthritis':12,
'(vertigo) Paroymsal  Positional Vertigo':13,'Acne':14,'Urinary tract infection':15,'Psoriasis':16,
'Impetigo':17}},inplace=True)

# print(df.head())

X= df[l1]
print(X)
y = df[["prognosis"]]
print(y)

np.ravel(y)

tr=pd.read_csv("static/Testing_Disease.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'Drug Reaction':2,
'Peptic ulcer diseae':3,'Migraine':4,'Paralysis (brain hemorrhage)':5,'Jaundice':6,'Chicken pox':7,'Common Cold':8,'Dimorphic hemmorhoids(piles)':9,
'Varicose veins':10,'Hypothyroidism':11,'Arthritis':12,
'(vertigo) Paroymsal  Positional Vertigo':13,'Acne':14,'Urinary tract infection':15,'Psoriasis':16,
'Impetigo':17}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)


gnb = GaussianNB()
gnb=gnb.fit(X,np.ravel(y))


y_pred=gnb.predict(X_test)
print("Accuracy",accuracy_score(y_test, y_pred))

with open('GausianNB_disease_model', 'wb') as files:
    pickle.dump(gnb, files)

clf=RandomForestClassifier()
clf=clf.fit(X,np.ravel(y))

with open('RandomForest_disease_model', 'wb') as files:
    pickle.dump(clf, files)

classifier=DecisionTreeClassifier()
classifier=classifier.fit(X,np.ravel(y))

with open('DecisionTreet_disease_model', 'wb') as files:
    pickle.dump(classifier, files)