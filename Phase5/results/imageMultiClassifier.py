import os
import sys
import subprocess
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
import re

folders=[]
value=[]
def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


def getValue(folder,precision):
	try:
		filename 		=folder+"/"+precision+"/results_comparation.txt"
		file  			=open(filename, "r") 
		read 			=0
		predictionValue =""
		oracleValue		=""
		localValue		=[]
		nPred			=0
		nShape			=0
		staticTime		=0
		staticMethod	="None"
		for line in file:
			line=line.strip()
			if(line=="Final Report:"):
				read=1
			if(read==1):
				if(line.find("With dynamic prediction of the algorithm:")!=-1):
					nPred =line.split(":")[1].split("|")[1].split("experiments")[0].strip()
					nShape=line.split(":")[1].split("|")[1].split("on")[1].strip()
					if(nPred!=nShape):
						predictionValue=0
					else:
						predictionValue=line.split(":")[1].split("|")[0].strip()
				elif(line.find("Best possible time:")!=-1):
					oracleValue		=line[len("Best possible time:"):].strip()
				elif(line.find("Best static method:")!=-1):
					staticTime		=line.split(":")[1].split("|")[0].strip()
					staticMethod	=line.split(":")[1].split("| with")[1].strip()
		if(predictionValue!="" and oracleValue!=""):
			localValue.append(folders[i])
			localValue.append(precision)
			localValue.append(predictionValue)
			localValue.append(oracleValue)
			localValue.append(staticTime)
			localValue.append(staticMethod)
			file.close()
			return localValue
	except Exception as e:
		print(e)		
		localValue		=[]
		localValue.append(folders[i])
		localValue.append(precision)
		localValue.append(0)
		localValue.append(0)
		localValue.append(0)
		localValue.append("None")
		return localValue

if(len(sys.argv)>2):
	title=sys.argv[1]
	for i in range(2,len(sys.argv)):
		folders.append(sys.argv[i])
else:
	print("No arguments, script terminated")

for i in range(0, len(folders)):

	readedValue		=getValue(folders[i],"default")
	value.append(readedValue)

	readedValue		=getValue(folders[i],"uint8")
	value.append(readedValue)


if(len(value)>1):
	print("Starting..")
else:
	print("No value collected, aborting")
	sys.exit()

#########################################
#Making value for the graph
predictedValue=[]
oracleValue=[]
staticValue=[]
labels=[]
for i in range(0,len(value)):
	print(value[i][0])
	print(value[i][1])
	print(value[i][2])
	print(value[i][3])
	print("\n\n")
	try:
		stringLabel=str(value[i][0].split("_")[1])
		if(value[i][0].split("_")>2):
			stringLabel+="_"+str(value[i][0].split("_")[2])
		stringLabel+=" "+str(value[i][1])+"\nPredicted: "+str(value[i][2])+"\nOracle: "+str(value[i][3])+"\n Best static: "+str(value[i][5])
		labels.append(stringLabel)
	except:
		labels.append(str(value[i][0])+" "+str(value[i][1])	+"\n"+str(value[i][2])+"\n"+str(value[i][3]))
	predictedValue.append(float(value[i][2]))
	oracleValue.append(float(value[i][3]))
	staticValue.append(float(value[i][4]))

print predictedValue
print oracleValue

ind = np.arange(len(predictedValue))
width = 0.23333
fig, ax = plt.subplots(figsize=(30,10))
rects1 = ax.bar(ind - width,	predictedValue, 	width,color='SkyBlue', 		label='Predicted',			bottom=0)
rects2 = ax.bar(ind ,			oracleValue, 		width,color='IndianRed', 	label='Oracle',				bottom=0)
rects3 = ax.bar(ind + width,	staticValue, 		width,color='green', 		label='Best static method',	bottom=0)

ax.set_ylabel('Execution Time (microseconds)')
ax.set_title(title)
ax.set_xticks(ind)
ax.set_xticklabels(labels)
ax.legend()

#autolabel(rects1, "left")
#autolabel(rects2, "right")

plt.savefig(title+".png", format='png', bbox_inches="tight")    
plt.savefig(title+".pdf", format='pdf', bbox_inches="tight")  
plt.cla()
plt.clf()
plt.close('all')
































