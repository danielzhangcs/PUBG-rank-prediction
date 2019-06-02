# Final Project Code(126a)
Team name: TODO
 
Team members: Kun Zhang, Shangyu Zhang, Qiang Guo, Wenxiao Xiao, Zijie Wu

### 1. file declaration

1.1. lightGBM.py

IO: read from original train and test data, finally write out submission_raw.csv

This is the lightGBM model.

1.2. pyTorchNN.py
 
IO: need internet io to install package from github and package repository. Also read from original train and test data, finally write out submission_raw.csv

This is the neural network model writen by library pyTorch.

1.3. adjust.py

IO: read intermediate result(usually submission_raw.csv) and test data, then adjust the result by group and write out submission_adjusted.csv.

This is a utility kernel to transform intermediate result to final submission result.

1.4. merge.py

IO: read from two kernels' results and take the average of them and write out submission result.

This is a utility kernel to merge two kernels' final result to a new submission result.

### 2. how to use the code to reproduce our result 0.0199

ATTENTION: All the code comes from our original kernel, which can ONLY run on kaggle kernel

2.1. create two kernels to fill code lightGBM.py and pyTorchNN.py respectively

2.2. run the two kernels(lightGBM's result score is 0.0238; pyTorchNN's result score is 0.0234) and adjust their results by a new kernel filled by adjust.py(lightGBM's result score becomes 0.0202; pyTorchNN's result score becomes 0.0203)

2.3. merge the two kernels' result(the score should be should be 0.0200)

2.4. adjust the result produced above and get score 0.0199
