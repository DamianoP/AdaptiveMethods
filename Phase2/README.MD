As a prerequisite, the CK software suite must be installed:
<br/>
please visit this link 
<br/>
https://github.com/ctuning/ck-nntest 
<br/>
and follow the steps
<br/>
Get the data.csv file (created in the Phase1) and import the file as a new dataset:<br/>

```
ck add ck-nntest:dataset:tensor-conv-test-myDataset --file=data.csv --tags=dataset,nntest,tensor-conv,conv,convolution
```
<br/>
Then prepare the shapes for the convolution with:
<br/>

```
ck run program:tool-prepare-dataset
```

You must select your dataset and wait for the preparation of the files.
<br/>
Once the dataset here are the steps to take.
<br/>
Before starting the new tests, I recommend removing any old experiments:

```
sudo ck rm local:experiment:* -f
```

<br/>
Run benchmarks with the following commands:

```
sudo ck run nntest:conv-armcl-opencl --dataset_uoa=tensor-conv-test-myDataset --iterations=1
sudo ck run nntest:directconv-armcl-opencl --dataset_uoa=tensor-conv-test-myDataset --iterations=1
sudo ck run nntest:winogradconv-armcl-opencl --dataset_uoa=tensor-conv-test-myDataset --iterations=1
```

<br/>

After completing the benchmarks, collect the data in zip files:

```
ck zip local:experiment:*-directconv-* --archive_name=myDatasetDirectconv.zip
ck zip local:experiment:*-conv-* --archive_name=myDatasetConv.zip
ck zip local:experiment:*-winogradconv-* --archive_name=myDatasetWinogradconv.zip
```

<br/>
Add zip files to the system repositories:

```
ck add repo:myDatasetDirectconv --zip=myDatasetDirectconv.zip --quiet
ck add repo:myDatasetConv --zip=myDatasetConv.zip --quiet
ck add repo:myDatasetWinogradconv --zip=myDatasetWinogradconv.zip --quiet
```

<br/>
Remove the experiments from the local storage:
<br/>

```
sudo ck rm local:experiment:* -f
```

<br/>
Run the experiments with precision "uint8":
<br/>

```
sudo ck run nntest:conv-armcl-opencl-uint8 --dataset_uoa=tensor-conv-test-myDataset --iterations=1
sudo ck run nntest:directconv-armcl-opencl-uint8 --dataset_uoa=tensor-conv-test-myDataset --iterations=1
```

After completing the benchmarks, collect the data in zip files:

```
ck zip local:experiment:*-directconv-* --archive_name=myDatasetDirectconvUINT8.zip
ck zip local:experiment:*-conv-* --archive_name=myDatasetConvUINT8.zip
```

Add zip files to the system repositories:

```
ck add repo:myDatasetDirectconvUINT8 --zip=myDatasetDirectconvUINT8.zip --quiet
ck add repo:myDatasetConvUINT8 --zip=myDatasetConvUINT8.zip --quiet
```

<br/>
Remove the experiments from the local storage:
<br/>

```
sudo ck rm local:experiment:* -f
```



After these steps you can go to the Phase3 section
<br/>





























