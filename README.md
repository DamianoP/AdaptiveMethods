# AdaptiveMethods
Adaptive methods for accelerating Deep Neural Networks on ARM GPU architectures
<br/>
This repository contains all the codes necessary to reproduce the experiments and the results obtained.
<br/>
The work has been enclosed in various folders with a name that describes the phase to be addressed.
<br/>
I recommend following the steps from the first to the last.
<br/><br/>
Phase1 explains how to create a dataset for experiments.
<br/><br/>
Phase2 explains how to perform the experiments and collect the results.
<br/><br/>
Phase3 explains how to read the results and get a .arff file
<br/><br/>
Phase4 explains how to get a model to use with machine learning techniques
<br/><br/>
Phase5 explains how to get the results of the analysis and contains some experiments carried out by me


<br/><br/>
<br/>
In these example images you can see the bar chart.
<br/>
The height of the bars indicates the running time.
<br/>
Please note that at the bottom of each bar is the number of successful experiments on the dataset taken into consideration.
<br/>
The predicted column indicates in brackets the number of experiments used for each technique: for example (3,7,9) it would indicate 3 experiments obtained with Conv, 7 with Directconv, 9 with Winograd.
<br/>
<img width="600" height="600" src="Phase5/results/MLP_Inception/default/img/global.png?raw=true">
<img width="600" height="600" src="Phase5/results/Alexnet_DecisionTree/default/img/global.png?raw=true">
