import os
import subprocess
import sys
if(len(sys.argv)>1):
	folder 		=sys.argv[1]
else:
	print("Error, the subprocess need to know the folder")

try:
	os.system("rm -r "+folder)
	os.system("mkdir "+folder)
	os.system("mkdir "+folder+"/default")
	os.system("mkdir "+folder+"/uint8")
	os.system("cp models/"+folder+".joblib "+folder+"/"+folder+".joblib")
	csvName=folder.split("_")[0]
	os.system("cp models/"+csvName+".csv "+folder+"/default/data.csv")
	os.system("cp models/"+csvName+".csv "+folder+"/uint8/data.csv")
	
	os.system("cp models/rankingDefault.txt "+folder+"/default/ranking.txt")
	os.system("cp models/rankingUint8.txt "+folder+"/uint8/ranking.txt")
	p=subprocess.Popen(["python","script.py",folder,"default"])
	p.wait()
except:
	print("script for "+folder+" with precision default fail the execution")
	
try:
	p=subprocess.Popen(["python","script.py",folder,"uint8"])
	p.wait()
except:	
	print("script for "+folder+" with precision uint8 fail the execution")

