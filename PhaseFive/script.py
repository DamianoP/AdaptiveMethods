import os
import sys
import json
import re
import IPython as ip
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib as mp
import seaborn as sb
import ck.kernel as ck
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import Image, display
np.set_printoptions(threshold='nan')
from joblib import load
def myPrint(string):
	print(string)
	newFile.write(string+"\n")


folder			="alexnetDecisionTree"	#raw_input("Name of folder: ")
fileName		="results"		#raw_input("Name of dataset: ")
precision		="default"
path			=folder+"/"+precision
shapeCSV		=path+"/data.csv"
rankingFile		=path+"/ranking.txt"

newFile			=open(path+"/"+fileName+"_comparation.txt","w")
accuracy		=0
tuner 			="false"
architecture	="midgard"
classifierName	="Decision tree"
images			=1 #1 for generate the images, 0 for generate only the text file
modelName=folder
myPrint("loading model...")
clf = load(folder+"/"+modelName+".joblib")
myPrint("model loaded !") 
	
#########
os.system("rm -r "+path+"/img")
os.system("mkdir "+path+"/img")
#########

#PREPROCESSING



if(tuner=="false"):
	tuner=0
else:
	tuner=1
if (precision=="default"):
	precision=0
else:
	precision=1
if(architecture=="midgard"):
	architecture=0
else:
	architecture=1

#LOADING DATA
myPrint("Loading data in memory..")
rankings = [line.rstrip('\n') for line in open(rankingFile)]
for i in range(0,len(rankings)):
	rankings[i]=eval(rankings[i])
myPrint("Ranking File loaded")

shapes = [line.rstrip('\n') for line in open(shapeCSV)]
myPrint("Shapes File loaded")
myPrint("Data loaded correctly")

predictedBest					=""
predicted						=0
localPredictedTime				=0
globalPredictedTime				=0

realBest						=""
localBestTime					=0
globalBestTime					=0

# TIME FOR THE CALCULATION OF ALL THE DATASET
globalTimeConv 					=0
globalTimeDirectConv 			=0
globalTimeWinogradConv			=0

# TIME FOR THE CALCULATION OF ALL THE SHAPE FOR EACH NETWORK
localTimeConv 					=0
localTimeDirectConv 			=0
localTimeWinogradConv			=0

# TIME FOR THE CALCULATION OF ALL THE SHAPE FOR EACH NETWORK
localTimePredictedConv 			=0
localTimePredictedDirectConv 	=0
localTimePredictedWinogradConv	=0
localPredConv					=0
localPredDirect					=0
localPredWinog					=0
# TIME FOR THE CALCULATION OF ALL THE SHAPE FOR EACH NETWORK
globalTimePredictedConv 		=0
globalTimePredictedDirectConv 	=0
globalTimePredictedWinogradConv	=0
globalPredConv					=0
globalPredDirect				=0
globalPredWinog					=0

#counter
nConv							=0
nWinograd						=0
nDirectConv						=0
nPredicted						=0
nShapes 						=0
nLocalPredicted					=0
nLocalConv						=0
nLocalDirectConv				=0
nLocalWinograd					=0
myPrint("Preprocessing ended")
#END PREPROCESSING
currentNetwork="_empty"
ax=""

def autolabel(rects, text):
	global ax
	rect=rects[0]
	height = rect.get_height()
	ax.text(rect.get_x() + (rect.get_width()/2)-0.05, 1.01*height,text)
	
def generateImage(imgName,time1,time2,time3,predictedConv,predictedDirectConv,predictedWinograd,
					stringPredicted,classifier,numConv,numDirect,numWinog,numPred,
					numPredConv,numPredDirect,numPredWinog):
	global ax
	imgName=str(imgName)
	xLabels=[]
	plt.figure(figsize=(10,10))
	plt.rcParams.update({'font.size': 14})
	if(numConv!="null"):
		convSTR="Conv"			+"\n"+str(numConv)		+" exp"
	else:
		convSTR="Conv"

	if(numDirect!="null"):
		directSTR="img2col + gemm"		+"\n"+str(numDirect)	+" exp"
	else:
		directSTR="img2col + gemm"

	if(numWinog!="null"):
		winogSTR="Winograd"	+"\n"+str(numWinog)	+" exp"
	else:
		winogSTR="Winograd"

	if(numPred!="null"):
		predicSTR="Predicted"		+"\n"+str(numPred)	+" exp \n"+"("+str(numPredConv)+", "+str(numPredDirect)+", "+str(numPredWinog)+")"
	else:
		predicSTR="Predicted"

	if(time1=="null"):
		time1=0
	if(time2=="null"):
		time2=0
	if(time3=="null"):
		time3=0
	if(predictedConv=="null"):
		predictedConv=0
	if(predictedDirectConv=="null"):
		predictedDirectConv=0
	if(predictedWinograd=="null"):
		predictedWinograd=0

	ind = np.arange(4)    # the x locations for the groups
	width = 0.35       # the width of the bars: can also be len(x) sequence

	#time3= 10000 # DA LEVARE ######################

	b1=[]
	b1.append(time1)
	b1.append(0)
	b1.append(0)
	b1.append(predictedConv)

	b2=[]
	b2.append(0)
	b2.append(time2)
	b2.append(0)
	b2.append(predictedDirectConv)

	b3=[]
	b3.append(0)
	b3.append(0)
	b3.append(time3)
	b3.append(predictedWinograd)

	bottomValue=np.array(b1)+np.array(b2)
	p1 = plt.bar(ind, b1, width)
	p2 = plt.bar(ind, b2, width,bottom=b1)
	p3 = plt.bar(ind, b3, width,bottom=bottomValue)

	plt.ylabel('Execution Time')
	plt.title(imgName)
	plt.xticks(ind, (convSTR, directSTR, winogSTR, predicSTR))
	plt.legend((p1[0], p2[0], p3[0]), ('Conv', 'Directconv','Winograd'),loc='upper center', bbox_to_anchor=(1,1.14), fancybox=True, shadow=True,ncol=1)
	
	'''
	h1 = p1.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
    '''

	plt.savefig(path+"/img/"+imgName+".png", format='png')    
	plt.cla()
	plt.clf()
	plt.close('all')



def middleReport():
	global localTimeConv,localTimeDirectConv,localTimeWinogradConv,localBestTime,predictedBest,localPredictedTime,classifierName,currentNetwork,images,nLocalConv,nLocalPredicted,nLocalWinograd,nLocalDirectConv,localTimePredictedConv,localTimePredictedDirectConv,localTimePredictedWinogradConv,localPredConv,localPredDirect,localPredWinog
	myPrint("Results:")
	myPrint("manual Conv time:"					+str(localTimeConv))
	myPrint("manual Directconv time:"			+str(localTimeDirectConv))
	myPrint("manual Winogradconv time:"			+str(localTimeWinogradConv))
	myPrint("best possible time:"				+str(localBestTime))
	myPrint("predicted time for the network:"	+str(localPredictedTime))
	myPrint("-----------------------------------------------------------")
	myPrint("\n \n \n")
	if (images==1):
		generateImage(currentNetwork,localTimeConv,localTimeDirectConv,localTimeWinogradConv,
			localTimePredictedConv,localTimePredictedDirectConv,localTimePredictedWinogradConv,"",
			classifierName,nLocalConv,nLocalDirectConv,nLocalWinograd,nLocalPredicted,localPredConv,localPredDirect,localPredWinog) # image for the shape

	localTimeConv 		 	=0
	localTimeWinogradConv	=0
	localTimeDirectConv		=0
	localBestTime 			=0
	localPredictedTime		=0
	predictedBest 			=""
	nLocalPredicted			=0
	nLocalConv				=0
	nLocalDirectConv		=0
	nLocalWinograd			=0
	localPredConv			=0
	localPredDirect			=0
	localPredWinog			=0
	localTimePredictedConv 			=0
	localTimePredictedDirectConv 	=0
	localTimePredictedWinogradConv	=0

##################
#MAIN
##################

for i in range(1,len(shapes)):
	shape=shapes[i].split(",")	
	if(len(shape[0])==0 and len(shape[1])==0 and len(shape[2])==0 ): #"skipping case : ',,,,,,,,' "
		continue
	if(len(shape[0])>0 and i>1 and currentNetwork!=shape[0]):
		if(currentNetwork=="_empty"):
			currentNetwork=shape[0]
		else:
			middleReport()
			currentNetwork=shape[0]
			myPrint("Analyzing "+shape[0])
	nShapes 		   +=1
	workingShape		=shape[3]+"-"+shape[1]+"-"+shape[2]+"-"+shape[4]+"-"+shape[5]+"-"+shape[6]+"-"+shape[7]
	workingShapeARFF	=shape[3]+","+shape[1]+","+shape[2]+","+shape[4]+","+shape[5]+","+shape[6]+","+shape[7]+","+str(tuner)+","+str(precision)+","+str(architecture)
	workingShapeARFF	=eval("[["+workingShapeARFF+"]]")
	predictedBest		=clf.predict(workingShapeARFF)[0]
	convTime			="null"
	directconvTime		="null"
	winogradconvTime	="null"
	predictedTimeShape	="null"
	finded				=False
	for j in range(0,len(rankings)):
		if(rankings[j][0][0]==workingShape):
			timeList=rankings[j][1]
			for h in range(0,len(rankings[j][1])):
				if(rankings[j][1][h][0][0]==predictedBest):
					predictedTimeShape		 =float(rankings[j][1][h][1][0])
					localPredictedTime		+=float(rankings[j][1][h][1][0])
					globalPredictedTime		+=float(rankings[j][1][h][1][0])
					nPredicted				+=1
					nLocalPredicted			+=1
					if(rankings[j][1][h][0][0]=="conv"):
						localTimePredictedConv 			+=float(rankings[j][1][h][1][0])
						globalTimePredictedConv 		+=float(rankings[j][1][h][1][0])
						globalPredConv 					+=1
						localPredConv 					+=1
					if(rankings[j][1][h][0][0]=="winogradconv"):
						localTimePredictedWinogradConv	+=float(rankings[j][1][h][1][0])
						globalTimePredictedWinogradConv +=float(rankings[j][1][h][1][0])
						globalPredWinog					+=1
						localPredWinog					+=1
					if(rankings[j][1][h][0][0]=="directconv"):
						localTimePredictedDirectConv	+=float(rankings[j][1][h][1][0])
						globalTimePredictedDirectConv 	+=float(rankings[j][1][h][1][0])
						globalPredDirect				+=1
						localPredDirect					+=1
				if(h==0):
					realBest 				 =rankings[j][1][h][0][0]
					localBestTime			+=float(rankings[j][1][h][1][0])
					globalBestTime			+=float(rankings[j][1][h][1][0])
				if(rankings[j][1][h][0][0]=="conv"):
					convTime 				 =float(rankings[j][1][h][1][0])
					localTimeConv	   		+=float(rankings[j][1][h][1][0])
					globalTimeConv	   		+=float(rankings[j][1][h][1][0])	
					nConv 					+=1	
					nLocalConv				+=1
					finded=True
				elif (rankings[j][1][h][0][0]=="winogradconv"):
					winogradconvTime 		 =float(rankings[j][1][h][1][0])
					localTimeWinogradConv	+=float(rankings[j][1][h][1][0])
					globalTimeWinogradConv	+=float(rankings[j][1][h][1][0])	
					nWinograd 				+=1
					nLocalWinograd			+=1
					finded=True
				elif (rankings[j][1][h][0][0]=="directconv"):
					directconvTime 		 	 =float(rankings[j][1][h][1][0])
					localTimeDirectConv		+=float(rankings[j][1][h][1][0])
					globalTimeDirectConv	+=float(rankings[j][1][h][1][0])	
					nDirectConv 			+=1
					nLocalDirectConv 		+=1
					finded=True
				else: 
					continue


			#rankings.remove(rankings[j])
			break
	if(finded==False):
		myPrint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
	myPrint("Analyzing: "+str(workingShape))
	myPrint("manual conv time: "			+str(convTime))
	myPrint("manual directconv time: "		+str(directconvTime))
	myPrint("manual winogradconvTime time: "+str(winogradconvTime))
	myPrint("predicted best: "				+str(predictedBest))
	myPrint("\n")
	cPred=0
	wPred=0
	dPred=0
	if(predictedBest=="conv"):
		cPred=convTime
	if(predictedBest=="winogradconv"):
		wPred=winogradconvTime
	if(predictedBest=="directconv"):
		dPred=directconvTime
	if (images==1):
		generateImage(workingShapeARFF,convTime,directconvTime,winogradconvTime,cPred,dPred,wPred,
			"",classifierName,"null","null","null","null",0,0,0) # image for the shape

	# SHAPE + time 1, time 2, time 3, time predicted, predicted method as string 
middleReport() #last shape
myPrint("\n")
myPrint("\n")
myPrint("-----------------------------------------------------------")
myPrint("\n")
myPrint("Final Report:")
myPrint("If you run all the dataset you will get this time:")
myPrint("Manual with conv:"							+str(globalTimeConv)		+ " | " + str(nConv) 		+" experiments successfully achieved on "+str(nShapes) )
myPrint("Manual with directconv:"					+str(globalTimeDirectConv)	+ " | " + str(nDirectConv) 	+" experiments successfully achieved on "+str(nShapes) )
myPrint("Manual with winogradconv:"					+str(globalTimeWinogradConv)+ " | " + str(nWinograd) 	+" experiments successfully achieved on "+str(nShapes) )
myPrint("With dynamic prediction of the algorithm:"	+str(globalPredictedTime)	+ " | " + str(nPredicted) 	+" experiments successfully achieved on "+str(nShapes) )
myPrint("Best possible time:"						+str(globalBestTime) )
if (images==1):
		generateImage("global",globalTimeConv,globalTimeDirectConv,globalTimeWinogradConv,globalTimePredictedConv,globalTimePredictedDirectConv,globalTimePredictedWinogradConv,"",classifierName,nConv,nDirectConv,nWinograd,nPredicted,globalPredConv,globalPredDirect,globalPredWinog)


myPrint("\n")
myPrint("Done!")
newFile.close()
















##################################################
######				CODE BACKUP				######
#X=[[]]
#X[0]=[48,35,35,5,64,1,2,0,0,0]
#print clf.predict(X)[0]
#return a+bprint("Working..")
##################################################
