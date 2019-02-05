import os

networkNames    =["Alexnet","inceptionV3","Mobilenets","ResNet","VGG-16","FCN-16s","Network-in-Network","GoogLeNet"]


models=os.listdir(".")

for i in range(0,len(models)):
	if(models[i].find("dataset_")!=-1):
		classifierName =models[i].split("_")[1]
		crossvalidation=models[i].split("_")[2]
		for j in range(0,len(networkNames)):
			os.system("cp "+models[i]+" "+networkNames[j]+"_"+classifierName+"_"+crossvalidation)