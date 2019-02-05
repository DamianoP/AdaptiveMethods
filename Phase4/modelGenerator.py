from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import linear_model
from joblib import dump, load
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy,scipy.io, scipy.io.arff
import pandas as pd
import sys
#### PREPROCESSING
clf=""
print("Loading dataset..")
if(len(sys.argv)>1):
	datasetName=sys.argv[1]
else:
	datasetName=raw_input("Dataset file (with extension): ")
dataset = loadarff(open(datasetName,'r'))
'''
dataset[0]['tuner'][dataset[0]['tuner']=='false']=0
dataset[0]['tuner'][dataset[0]['tuner']=='true']=1
dataset[0]['precision'][dataset[0]['precision']=='default']=0
dataset[0]['precision'][dataset[0]['precision']=='uint8']=1
dataset[0]['architecture'][dataset[0]['architecture']=='midgard']=0
dataset[0]['architecture'][dataset[0]['architecture']=='byfrost']=1
'''
train = np.array(dataset[0][['tensor_1', 'tensor_2', 'tensor_3', 'tensor_4', 'tensor_5','tensor_6','tensor_7','tuner','precision','architecture']],dtype=[('tensor_1', float),('tensor_2', float),('tensor_3', float),		('tensor_4', float),('tensor_5', float),('tensor_6', float),('tensor_7', float),('tuner',float),('precision',float),('architecture',float)])
target = np.array(dataset[0]['class'])
X=train.copy()
X = X.view((float, len(X.dtype.names)))
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.20, random_state=5, shuffle=True)
print("Dataset loaded")
#### END PREPROCESSING

if(len(sys.argv)>2):
	selection1=sys.argv[2]
else:
	print("Please digit wich classifier you want to use:")
	print("1 for Random Forest")
	print("2 for Multi layer Perceptron")
	print("3 for Decision Tree")
	#print("4 for Bayesian Tree")
	print("4 for Logistic Regression")
	print("5 for Linear Regression")
	print("6 for Naive bayes")
	print("7 for Support Vector Machine")
	selection1=raw_input("Classifier n.")
print("\n\n")

if(len(sys.argv)>3):
	selection2=sys.argv[3]
else:
	print("Please digit wich metod do you want to use for create the model: ")
	print("1 for use all the dataset for the training")
	print("2 for use cross validation 80 / 20")
	selection2=raw_input("Method n.")

if(selection1=="1"):
	clf = RandomForestClassifier(n_estimators=300, max_depth=50,random_state=0)
	modelName="RandomForest"
elif(selection1=="2"):
	print("\n")
	print("Please wait...")
	clf = MLPClassifier(solver='lbfgs', activation="tanh" ,hidden_layer_sizes=(50), random_state=1,max_iter=999999999,shuffle=True )
	modelName="MLP"
elif(selection1=="3"):
	clf = DecisionTreeClassifier()
	modelName="DecisionTree"
elif(selection1=="4"):
	clf = LogisticRegression(random_state=0, solver='newton-cg', C=1000000, multi_class='multinomial',max_iter=100000)
	modelName="Logistic"
elif(selection1=="5"):
	dataset[0]['class'][dataset[0]['class']=='conv']=0
	dataset[0]['class'][dataset[0]['class']=='directconv']=1000
	dataset[0]['class'][dataset[0]['class']=='winogradconv']=-1000
	target = np.array(dataset[0]['class'],dtype=[('class', float)])
	targetTemp=[]	
	for i in range(0, target.size):
		targetTemp.append(target[i][0])
	target=targetTemp
	X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=5, shuffle=True)
	clf = linear_model.LinearRegression()	
	modelName="LinearRegression"
elif(selection1=="6"):
	clf = GaussianNB(priors=None, var_smoothing=180e-9)
	modelName="Bayesian"
elif(selection1=="7"):
	clf = LinearSVC(random_state=0,tol=1, max_iter=100000)
	modelName="SVM"
else:
	print("You have entered a wrong input")
	quit()
print clf
if(selection2=="1"):
	print("Testing without cross validation")
	clf.fit(X, target)
	print("Model created")
	print ("Score: "+str(clf.score(X, target)))
	predicted = clf.predict(X)
	modelName=modelName+"_ALL"

elif(selection2=="2"):
	print("Testing with cross validation")
	clf.fit(X_train, y_train)
	print("Model created")
	scores = cross_val_score(clf, X_test, y_test, cv=2)
	print ("Score:  %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))	
	predicted = clf.predict(X_test)	
	target=y_test
	modelName=modelName+"_CV"

else:
	print("You have entered a wrong input")
	quit()

if(selection1!="5"):
	print("Confusion matrix:")
	print confusion_matrix(target, predicted, labels=['conv', 'directconv', 'winogradconv'])

modelName=datasetName.split(".")[0]+"_"+modelName+".joblib"
dump(clf, modelName)
print(modelName+" created")

