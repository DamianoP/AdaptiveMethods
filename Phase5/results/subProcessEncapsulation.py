import os
import subprocess
import sys
if(len(sys.argv)>1):
	folder 		=sys.argv[1]
else:
	print("Error, the subprocess need to know the folder")

try:
	p=subprocess.Popen(["python","script.py",folder,"default"])
	p.wait()
except:
	print("script for "+folder+" with precision default fail the execution")
	
try:
	p=subprocess.Popen(["python","script.py",folder,"uint8"])
	p.wait()
except:	
	print("script for "+folder+" with precision uint8 fail the execution")

