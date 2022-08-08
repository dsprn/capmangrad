# CapmanGrad

## What is it?
CapmanGrad is yet another implementation of [micrograd](https://github.com/karpathy/micrograd)  done almost entirely from memory, as a personal learning project. It has some added functionality, namely Cross Validation, which is used to compute the best possible L2 Regularization lambda hyperparameter over a range of possible values. It's more of a mean to learn as much something about the subject than a production piece of software. Things could be added in the future as a further exercize about the topic.
Feel free to use it anyway you see fit.

## Why is it called like that?
I thought about calling it something along the lines of HomeworkGrad but then my resources hungry implementation of Cross Validation, coupled with my weird sense of humour, changed my mind. Besides, CapmanGrad, aside from being a Pacman anagram with a Grad added to it, is just a better name, even for a learning project.

## Usage
Right now the best way to use it is to run the following command from your terminal emulator (please note you should first create a python virtual environment and install all the needed requirements for it to work properly)

```python
python demo.py
```

This will run the demo file which will start tuning the L2 regularization lambda hyperparameter needed to run the regularization algorithm on the model at a later stage.
It will also save the model to a json file once the training's done and will render the whole associated computational graph, saving it to an SVG file.

Following is the output produced by the demo.py script

```
==> Computing L2 lambda penalty hyperparameter...
==> Using Cross Validation to look for best hyperparameter in values ranging from 0.0 to 0.0095
hyperparameter:0.0, loss:0.39467498521359856, accuracy:81.11%
hyperparameter:0.0005, loss:0.40636039418523195, accuracy:88.89%
hyperparameter:0.001, loss:0.29609627339651784, accuracy:87.78%
hyperparameter:0.0015, loss:0.4223004997139719, accuracy:84.44%
hyperparameter:0.002, loss:0.3969131730817356, accuracy:80.0%
hyperparameter:0.0025, loss:0.36046924719721685, accuracy:82.22%
hyperparameter:0.003, loss:0.35387977916704777, accuracy:87.78%
hyperparameter:0.0035, loss:0.5923888504763057, accuracy:73.33%
hyperparameter:0.004, loss:0.555335743517535, accuracy:76.67%
hyperparameter:0.0045, loss:0.6357716576549469, accuracy:72.22%
hyperparameter:0.005, loss:0.4130479851334161, accuracy:84.44%
hyperparameter:0.0055, loss:0.5480736490487311, accuracy:75.56%
hyperparameter:0.006, loss:0.7659157615316966, accuracy:65.56%
hyperparameter:0.0065, loss:0.5140332467054155, accuracy:78.89%
hyperparameter:0.007, loss:0.5109978596737291, accuracy:80.0%
hyperparameter:0.0075, loss:0.489320964770775, accuracy:81.11%
hyperparameter:0.008, loss:0.5833758764693341, accuracy:77.78%
hyperparameter:0.0085, loss:0.527656094197141, accuracy:75.56%
hyperparameter:0.009, loss:0.5229692044685738, accuracy:76.67%
hyperparameter:0.0095, loss:0.5810701605023351, accuracy:70.0%
==> Best hyperparameter in given range: 0.0005
==> Start training the model...
epoch:1, loss:1.0383517832371345, accuracy:56.57%
epoch:2, loss:2.9752721544946024, accuracy:49.49%
epoch:3, loss:0.7827231317175338, accuracy:71.72%
epoch:4, loss:0.5710326293964952, accuracy:83.84%
epoch:5, loss:0.38510803081557576, accuracy:86.87%
epoch:6, loss:0.32358032764844, accuracy:87.88%
epoch:7, loss:0.2990345227479948, accuracy:87.88%
epoch:8, loss:0.2912119199620543, accuracy:89.9%
epoch:9, loss:0.2640468238786426, accuracy:89.9%
epoch:10, loss:0.2511791461395704, accuracy:89.9%
epoch:11, loss:0.24597307716844546, accuracy:90.91%
epoch:12, loss:0.2546784393964371, accuracy:91.92%
epoch:13, loss:0.2623325860910245, accuracy:89.9%
epoch:14, loss:0.309474545306686, accuracy:90.91%
epoch:15, loss:0.3485080729057618, accuracy:89.9%
epoch:16, loss:0.3647356408281876, accuracy:89.9%
epoch:17, loss:0.25755125316180355, accuracy:89.9%
epoch:18, loss:0.2219060320793477, accuracy:91.92%
epoch:19, loss:0.20909145006084423, accuracy:91.92%
epoch:20, loss:0.19808939540974177, accuracy:93.94%
epoch:21, loss:0.1884755055172281, accuracy:93.94%
epoch:22, loss:0.17649054730775096, accuracy:93.94%
epoch:23, loss:0.1662745475852273, accuracy:93.94%
epoch:24, loss:0.169968798256624, accuracy:94.95%
epoch:25, loss:0.21788847473271672, accuracy:94.95%
epoch:26, loss:0.38804658843983314, accuracy:88.89%
epoch:27, loss:0.2564025535037994, accuracy:91.92%
epoch:28, loss:0.16546998394189774, accuracy:94.95%
epoch:29, loss:0.14285590015888971, accuracy:95.96%
epoch:30, loss:0.1311437970236127, accuracy:96.97%
epoch:31, loss:0.11967159011786477, accuracy:97.98%
epoch:32, loss:0.11479667044223904, accuracy:96.97%
epoch:33, loss:0.14072880737157784, accuracy:96.97%
epoch:34, loss:0.457347491652129, accuracy:85.86%
epoch:35, loss:0.28457522430459237, accuracy:90.91%
epoch:36, loss:0.15589873073001215, accuracy:95.96%
epoch:37, loss:0.12064211733912727, accuracy:96.97%
epoch:38, loss:0.10867339592347311, accuracy:96.97%
epoch:39, loss:0.1011280460641237, accuracy:97.98%
epoch:40, loss:0.10192151743181817, accuracy:97.98%
epoch:41, loss:0.12786172569240634, accuracy:95.96%
epoch:42, loss:0.09909266685116283, accuracy:97.98%
epoch:43, loss:0.11787476718684459, accuracy:95.96%
epoch:44, loss:0.08985492674448224, accuracy:97.98%
epoch:45, loss:0.10636101563334498, accuracy:97.98%
epoch:46, loss:0.08138602306899548, accuracy:98.99%
epoch:47, loss:0.07712657611978478, accuracy:98.99%
epoch:48, loss:0.08345894436269369, accuracy:98.99%
epoch:49, loss:0.09258574446057584, accuracy:98.99%
epoch:50, loss:0.20291473985263203, accuracy:94.95%
epoch:51, loss:0.21995561562585056, accuracy:94.95%
epoch:52, loss:0.36253491171123636, accuracy:90.91%
epoch:53, loss:0.18279397022637578, accuracy:94.95%
epoch:54, loss:0.11161264580990585, accuracy:96.97%
epoch:55, loss:0.08739963733460035, accuracy:98.99%
epoch:56, loss:0.08443541238093166, accuracy:98.99%
epoch:57, loss:0.08194466404308966, accuracy:100.0%
==> DONE
==> Exporting model to JSON file... DONE
==> Using model on single datapoint to render the computational graph... DONE
```

Please keep in mind that  this implementation of the cross validation algorithm could take quite a while.

## Tests
To run the tests associated with this project you should first install the pytest framework and, like it happens with micrograd, pytorch (look at the instructions on the project's website) and then you should be able to run the following command in your terminal emulator without any issues

```python
python -m pytest
```
