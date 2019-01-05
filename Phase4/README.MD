Take the .arff file (obtained in Phase3) and put it in the same folder of the script "modelGenerator.py".
<br/>
Simply run the script with:

```
python modelGenerator.py
```

<br/><br/><br/>
The purge.py script take in input two files.
<br/>
The .arff file and the .csv file with the same format as the file used to prepare the datasets.
<br/>
The script generates a new .arff file containing all the values of the .arff file in input with the exception of the tensors present in the .csv.
<br/>I used this file in the following phases to validate the percentage of recognition of the machine learning techniques implemented by me.
<br/>For example, in this way I could test with what precision the neural networks created by me can recognize "InceptionV3" starting from a model that does not contain any reference of those tensors.
You can run the purge.py with:

```
python purge.py
```
