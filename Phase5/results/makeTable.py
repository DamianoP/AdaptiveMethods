import IPython as ip
import pandas as pd
import matplotlib
from matplotlib.patches import Rectangle
matplotlib.use('Agg')
import numpy as np
import matplotlib as mp
import seaborn as sb
import ck.kernel as ck
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import Image, display
np.set_printoptions(threshold='nan')
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def getValue(folder,precision):
  results ="null"
  try:
    accuracy="err"
    speedUp ="err"
    error   =0
    filename    =folder+"/"+precision+"/results_comparation.txt"
    file        =open(filename, "r") 
    read      =0
    for line in file:
      line=line.strip()
      if(line=="Final Report:"):
        read=1
      if(read==1):
        if(line.find("Accuracy:")!=-1):
          accuracy=line.split(":")[1].split("|")[0].strip()    
        elif(line.find("SpeedUp:")!=-1):
          speedUp =line.split(":")[1].split("|")[0].strip()  
        elif(line.find("With dynamic prediction of the algorithm:")!=-1): 
          nPred =line.split(":")[1].split("|")[1].split("experiments")[0].strip()
          nShape=line.split(":")[1].split("|")[1].split("on")[1].strip()
          if(nPred!=nShape):
            error=1
    if(error==0):
      results="A="+accuracy+" S="+speedUp 
    else:
      results="A="+accuracy+" S="+speedUp+"**" 
  except Exception as e:
    print(e)        
  return results

def printFigure(data,classifierNames,title,precision):
  fig, ax = plt.subplots()
  matplotlib.rcParams.update({'font.size': 15})
  fig.set_size_inches(18.5, 5.5)
  # hide axes
  #fig.patch.set_visible(False)
  ax.axis('off')
  ax.axis('tight')
  #np.random.randn(10, 6)
  df = pd.DataFrame(data, columns=classifierNames)

  ax.table(cellText=df.values, colLabels=df.columns, loc='center',cellLoc='center').scale(1, 2)

  fig.tight_layout()
  plt.text(-0.023, -0.06,"** the results with the asterisk show some incomplete results\nfor example prediction suggested an algorithm that could not be used")
  plt.title("Accuracy\n "+title+" with precision "+precision+"\nA=accuracy, S=performance increase",y=0.83)
  plt.savefig(title+"_"+precision+".png", format='png')  
  plt.savefig(title+"_"+precision+".pdf", format='pdf')  
  print("saved image "+title+"_"+precision+".png")  
  plt.cla()
  plt.clf()
  plt.close('all')

def mainFunction(precision,networkNames,classifierNames,title):
  value=[]
  for i in range(0,len(networkNames)):
    localValue=[]
    localValue.append(networkNames[i])
    for j in range(1,len(classifierNames)):
      folderName=networkNames[i]+"_"+classifierNames[j]
      localValue.append(getValue(folderName,precision))
    value.append(localValue)
  printFigure(value,classifierNames,title,precision)

networkNames    =["Alexnet","Inception","Mobilenets160","ResNet","VGG-16","FCN-16s","Network-in-Network"]
classifierNames =["Network Name","Bayesian","SVM","MLP","DecisionTree","RandomForest"]
title="firefly_ALL"#raw_input("Image title: ")
mainFunction("default",networkNames,classifierNames,title)
mainFunction("uint8",networkNames,classifierNames,title)

networkNames    =["Alexnet_CV","Inception_CV","Mobilenets160_CV","ResNet_CV","VGG-16_CV","FCN-16s_CV","Network-in-Network_CV"]
classifierNames =["Network Name","Bayesian","SVM","MLP","DecisionTree","RandomForest"]
title="firefly_CV"#raw_input("Image title: ")
mainFunction("default",networkNames,classifierNames,title)
mainFunction("uint8",networkNames,classifierNames,title)









