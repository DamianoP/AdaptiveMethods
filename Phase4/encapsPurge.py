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


for i in range(0,len(files)):	
	if(files[i].find(".csv")!=-1):
		p=subprocess.Popen(["python","purge.py",files[i].split(".csv")[0]])
		processes.append(p)		
		processes=waitAndClean(processes)
	else:
		print("Skipping "+files[i])


for p in processes:
	p.wait()


print("All processed finished")