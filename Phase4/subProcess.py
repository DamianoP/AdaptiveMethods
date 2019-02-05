import os
import subprocess
import sys
if(len(sys.argv)>3):
	arff 		=sys.argv[1]
	choice1 	=sys.argv[2]
	choice2		=sys.argv[3]
else:
	print("Error, the subprocess need to know the folder")

try:
	print arff
	print choice1
	print choice2
	p=subprocess.Popen(["python","modelGenerator.py",arff,choice1,choice2])
	p.wait()
except:
	print("Error")
