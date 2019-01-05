#fileName=raw_input("Name of folder: ")
fileName="data"
print("Creating file: "+fileName+".csv")
file=open(fileName+".csv","w")
file.write(",W,H,C_IN,KERNEL_SIZE,C_OUT (filters count),STRIDE,PAD,,Layer name\n")

print("The script need some information")

#Parameters
W_MIN			=int(raw_input("Please enter the minimum width (suggested 7):")) #7
H_MIN			=1
C_IN_MIN		=int(raw_input("Please enter the minimum value for C_IN  (suggested 256): ")) #256
KERNEL_SIZE_MIN	=int(raw_input("Please enter the minimum value for Kernel Size  (suggested 1): ")) #1
C_OUT_MIN		=int(raw_input("Please enter the minimum value for C_OUT  (suggested 256): "))#256
STRIDE_MIN		=int(raw_input("Please enter the minimum value for Stride  (suggested 1): ")) #1
PAD_MIN			=int(raw_input("Please enter the minimum value for Padding  (suggested 1): ")) #1

W_MAX			=int(raw_input("Please enter the maximum width  (suggested 7): ")) #7
C_MAX 			=int(raw_input("Please enter the maximum value for C_IN  (suggested 2048): ")) #2048 
KERNEL_SIZE_MAX	=int(raw_input("Please enter the maximum value for Kernel Size  (suggested 9): ")) #11 #invece di 11
C_OUT_MAX 		=int(raw_input("Please enter the maximum value for C_OUT  (suggested 1024): ")) #1024 # invece di 512
STRIDE_MAX 		=int(raw_input("Please enter the maximum value for Stride  (suggested 1): ")) #1
PAD_MAX 		=int(raw_input("Please enter the maximum value for Padding  (suggested 1): ")) #1


LAYER_NAME=1
W=W_MIN
string=""
while W<=W_MAX:	
	H=W
	C_IN=C_IN_MIN
	while C_IN<=C_MAX:
		KERNEL_SIZE=KERNEL_SIZE_MIN
		while KERNEL_SIZE<=KERNEL_SIZE_MAX:
			C_OUT=C_OUT_MIN
			while C_OUT<=C_OUT_MAX:
				STRIDE=STRIDE_MIN
				while STRIDE<=STRIDE_MAX:
					PAD=PAD_MIN
					while PAD<=PAD_MAX:
						string=","+str(W)+","+str(H)+","+str(C_IN)+","+str(KERNEL_SIZE)+","+str(C_OUT)+","+str(STRIDE)+","+str(PAD)+","+","+"Test "+str(LAYER_NAME)+"\n"
						file.write(string)
						"""
						if (LAYER_NAME%100000)==1:
							print(LAYER_NAME)
						"""
						LAYER_NAME+=1
						PAD+=1
					STRIDE+=1
				C_OUT*=2
			KERNEL_SIZE+=2
		if(C_IN==3):
			C_IN=8
		else:
			C_IN*=2
	if(W==7):
		W=16
	else:
		W*=2

print("Generated "+str(LAYER_NAME)+" tests")
file.close()
