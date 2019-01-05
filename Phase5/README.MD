The results of my experiments are in the results folder.
<br/>
If you want to run the Phase5 manually create a folder and give it a name.
<br/>Insert the template file created in the previous step into the folder you created.
<br/>The model must be called the folder.
<br/>Inside the folder creates two sub folders: default, uint8.
<br/>Enter two files in the respective folders, the ranking.txt file and the data.csv file
<br/>Then you can run the script with the command:

```
python script.py
```


<br/>
<br/>
Images with bar charts will be automatically generated and a report file called comparation.txt will be created.
<br/>
<br/>The script can also be launched with the parameters entered from the command line.
<br/>The first parameter must be the folder to be processed.
<br/>The second parameter must be precision (subfolder name, "uint8" or "default")
<br/>The third parameter is optional, usually it must be the precision to work as soon as the processing of the first sub-folder is finished.
<br/>Here is an example to work the bayesianCV folder with uint8 precision:

```
python script.py bayesianCV uint8
```


<br/>Here is an example to work the bayesianCV folder with precision default and uint8:

```
python script.py bayesianCV default uint8
```

<br/><br/>
The python script encapsulation.py automates the generation of the report for all the folders in the current folder.
<br/>To run it use:

```
python incapsulation.py
```
