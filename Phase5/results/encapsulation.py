import os
import subprocess
import time
import multiprocessing
cpuCores=multiprocessing.cpu_count()
models=os.listdir("./models")
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





for i in range(0,len(models)):	
	if(models[i].find(".joblib")!=-1):
		model=models[i].split(".joblib")[0]
		processes=waitAndClean(processes)
		p=subprocess.Popen(["python","subProcessEncapsulation.py",model])
		processes.append(p)
	else:
		print("Skipping "+models[i])


for p in processes:
	p.wait()

p=subprocess.Popen(["python","encapsImage.py"])
processes.append(p)
p=subprocess.Popen(["python","makeTable.py"])
processes.append(p)

for p in processes:
	p.wait()

print("All processed finished")
