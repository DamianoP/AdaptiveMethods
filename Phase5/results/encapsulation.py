import os
import subprocess
import time
import multiprocessing
cpuCores=multiprocessing.cpu_count()
folders=os.listdir(".")
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





for i in range(0,len(folders)):	
	processes=waitAndClean(processes)
	if(os.path.isdir(folders[i])):
		p=subprocess.Popen(["python","subProcessEncapsulation.py",folders[i]])
		processes.append(p)
	else:
		print("Skipping "+folders[i])


for p in processes:
	p.wait()


print("All processed finished")
