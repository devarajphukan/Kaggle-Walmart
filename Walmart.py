import pandas as pd
import numpy as np
import re, nltk        
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import *
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

train_data_df = pd.read_csv('train.csv',delimiter=',',header = None)
test_data_df = pd.read_csv('test.csv',header = None ,delimiter=",")

train_data_df.columns = ["TripType","VisitNumber","Weekday","Upc","ScanCount","DepartmentDescription","FinelineNumber"]
test_data_df.columns = ["VisitNumber","Weekday","Upc","ScanCount","DepartmentDescription","FinelineNumber"]

train_data_df = train_data_df.fillna(0)
test_data_df = test_data_df.fillna(0)

train_data_df1 = train_data_df.drop('TripType', 1)
train_data_df1 = train_data_df1.drop('Weekday', 1)
train_data_df1 = train_data_df1.drop('DepartmentDescription', 1)
train_data_df1 = np.array(train_data_df1)

test_data_df1 = test_data_df.drop('Weekday',1)
test_data_df1 = test_data_df1.drop('DepartmentDescription',1)

"""
X_train, X_test, y_train, y_test  = train_test_split(train_data_df1, train_data_df.TripType, random_state=2)
#my_model = LinearSVC(penalty = 'l2',dual = True,C=0.7,loss='hinge')
my_model = LogisticRegression(penalty = 'l1')
my_model = my_model.fit(X=X_train, y=y_train)
test_pred = my_model.predict(X_test)
print classification_report(test_pred,y_test)
"""

my_model = LogisticRegression(penalty = 'l1')
my_model = my_model.fit(X=train_data_df1, y=train_data_df.TripType)
test_pred = my_model.predict(test_data_df1)

spl = []
for i in range(len(test_pred)) :
	spl.append(i)

rows = (((test_data_df1.VisitNumber).tolist())[-1]) + 1
results = np.zeros((rows,45))

for Vnum, Trtype in zip(test_data_df.VisitNumber[spl], test_pred[spl]) :
	#print Vnum,"-->",Trtype,"\n"
	results[Vnum][Trtype] += float(1)

fw = open("TotalResult.txt","w")
for i in range(rows) :
	res = str(i) 
	
	for j in range(45) :
		res += "," + str(int(results[i][j]))
	
	fw.write(res+"\n")			
