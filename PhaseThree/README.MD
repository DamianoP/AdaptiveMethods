With this script you can read the experiments collected in the PhaseTwo and get some files.
<br/>
The first file is an .arff file that is compatible with Weka and scikit (python).
<br/>
It also generates a second file (ranking) where each row is a list.
<br/>
The first element is the tensor.
<br/>
The second element is an ordered list of pairs (algorithm name, execution time)

Simply run the script with 

```
python readExp.py
```
