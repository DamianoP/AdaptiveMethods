import os
import subprocess
import time
import multiprocessing
cpuCores=multiprocessing.cpu_count()
files=os.listdir(".")
processes = []

def waitAndClean(processes):
	removed=0
	print("waitAndClean: "+str(len(processes)))
	localProcesses=processes
	while(len(localProcesses)==cpuCores):
		print("localProcesses: "+str(len(processes)))
		print("processes: "+str(len(processes)))
		for p in processes:
			print p.poll()
			if p.returncode != None:
				localProcesses.remove(p)
				removed=1
		if(removed==0):
			time.sleep(2)
	return localProcesses



classifier=["1","2","3","6","7"]
crossValidation=["1","2"]

for i in range(0,len(files)):	
	if(files[i].find(".arff")!=-1):
		for h in range(0,len(classifier)):
			for z in range(0,len(crossValidation)):
				p=subprocess.Popen(["python","subProcess.py",files[i],classifier[h],crossValidation[z]])
				processes.append(p)		
				processes=waitAndClean(processes)
	else:
		print("Skipping "+files[i])


for p in processes:
	p.wait()


print("All processed finished")
