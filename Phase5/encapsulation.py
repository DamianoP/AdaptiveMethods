import os
import subprocess

folders=os.listdir(".")

processes = []
for i in range(0,len(folders)):
	p=subprocess.Popen(["python","script.py",folders[i],"default"])
	processes.append(p)

for p in processes:
	p.wait()

processes = []
for i in range(0,len(folders)):
	p=subprocess.Popen(["python","script.py",folders[i],"uint8"])
	processes.append(p)

for p in processes:
	p.wait()

print("All processed finished")
