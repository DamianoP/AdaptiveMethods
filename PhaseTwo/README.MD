As a prerequisite, the CK software suite must be installed:
<br/>
please visit this link 
<br/>
https://github.com/ctuning/ck-nntest 
<br/>
and follow the steps
<br/>
Get the data.csv file (created in the PhaseOne) and import the file as a new dataset:<br/>

```
ck add ck-nntest:dataset:tensor-conv-test-myDataset --file=data.csv --tags=dataset,nntest,tensor-conv,conv,convolution
```
<br/>
Then prepare the shapes for the convolution with:
<br/>

```
ck run program:tool-prepare-dataset
```

You must select your dataset and waith for the preparation of the files.
<br/>


