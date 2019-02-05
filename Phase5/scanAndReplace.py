import os
import sys

rankDefault ="rankingDefault.txt"
rankUint8 	="rankingUint8.txt"
#dataCSV 	="VGG-16.csv"
def scanForFile(path):
	files=os.listdir(path)
	for i in range(0,len(files)):	
		try:
			fileAndPath=path+"/"+files[i]
			if(files[i]=="ranking.txt"):	
				if(fileAndPath.find("uint8")!=-1):
					#os.system("rm -r "+path+"/img")
					print("cp " + rankUint8 	+" "+fileAndPath)
					os.system("cp " + rankUint8 	+" "+fileAndPath)
				else:
					print("cp " + rankDefault	+" "+fileAndPath)
					os.system("cp " + rankDefault	+" "+fileAndPath)
			#elif(files[i]=="data.csv" and fileAndPath.find("VGG-16")!=-1):
			#	os.system("cp " + dataCSV	+" "+fileAndPath)
			else:
				scanForFile(path+"/"+files[i])
		except:
			continue
scanForFile(".")