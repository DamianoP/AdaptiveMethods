import sys

fileARFF	="dataset.arff"
if(len(sys.argv)>1):
	dataCSV 	=sys.argv[1]
else:
	dataCSV		=raw_input("Name of csv file: ")
purgedARFF	=open(dataCSV+".arff","w")
dataCSV 	=dataCSV+".csv"

purgedARFF.write("@RELATION convolution\n")
purgedARFF.write("@ATTRIBUTE tensor_1 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_2 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_3 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_4 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_5 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_6 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tensor_7 INTEGER\n")
purgedARFF.write("@ATTRIBUTE tuner INTEGER\n")
purgedARFF.write("@ATTRIBUTE precision INTEGER\n")
purgedARFF.write("@ATTRIBUTE architecture INTEGER\n")
purgedARFF.write("@ATTRIBUTE class {conv,directconv,winogradconv}\n")
purgedARFF.write("@DATA\n")
arffLines = [line.rstrip('\n') for line in open(fileARFF)]
csvLines  = [line.rstrip('\n') for line in open(dataCSV)]
for i in range(1,len(arffLines)):
	finded 		=0
	arffelement	=arffLines[i].split(",")
	if(len(arffelement)<4):
		continue
	row1		=","+arffelement[1]+","+arffelement[2]+","+arffelement[0]+","+arffelement[3]+","+arffelement[4]+","+arffelement[5]+","+arffelement[6]+","
	
	for j in range(1,len(csvLines)):
		try:
			csvElement	=csvLines[j].split(",")
			row2 		=","+csvElement[1]+","+csvElement[2]+","+csvElement[3]+","+csvElement[4]+","+csvElement[5]+","+csvElement[6]+","+csvElement[7]+","
			if(row1==row2):
				print("Removed: "+csvLines[j])
				finded=1
				#csvLines.remove(csvLines[j])
		except:
			print("Skipped: "+str(j))
	if(finded==0):
		purgedARFF.write(arffLines[i]+"\n")
print("result file: datasetWithout_"+dataCSV+".arff")
print("Done")