# import random
import numpy as np
from sklearn.datasets import make_moons
from capmangrad.xval import xval
from capmangrad.utils import computational_graph, MSE, SVM_maxmargin, L2, data_ratio, ANSWER
from capmangrad.engine import Value
from capmangrad.nn import Neuron, Layer, Model


# TODO: make this an argument from terminal
debug_mode = True

# initial dataset
if debug_mode:
    # if in debug_mode then use the random_state to always get the same data
    # also useful in cross validation as it freezes the inputs while looking for the best hyperpar
    dataset_vals, dataset_labels = make_moons(n_samples=100, noise=0.1, random_state=ANSWER)
else:
    dataset_vals, dataset_labels = make_moons(n_samples=100, noise=0.1)
# make the labels 1 or -1 instead of 1 and 0 (for binary accuracy computation)
dataset_labels = dataset_labels*2 - 1

# creating training/testing data/labels with a default ratio of 0.8/0.2
training_data, testing_data = data_ratio(dataset_vals, perc=0.99)
training_labels, testing_labels = data_ratio(dataset_labels, perc=0.99)
arch = [2, 16, 16, 1]
epoch = 1
alpha = 1.0 - 0.9*epoch/100

# L2 penalty hyperparameter
print('==> Computing L2 lambda penalty hyperparameter...')
# cross validation to find optimal L2 lambda value based on data
xv = xval(training_data, training_labels, arch, np.arange(0.0, 0.01, 0.0005), alpha, SVM_maxmargin, debug_mode, 10)
L2_lambda = xv()

# train on entire dataset (do not split in training and testing)
inputs = [list(map(Value, i)) for i in training_data]
# create the model (with random weights, not like the ones used in xval)
model = Model(arch[0], arch[1:])
# print(f"MODEL PARAMS SUM:{sum(model.params())}")
losses = None
acc = None

print("==> Start training the model...")
while not acc or acc<=99.0:
    model.zero_grad()
    # forward pass
    predictions = list(map(model, inputs))
    losses = list(map(SVM_maxmargin, predictions, training_labels))
    loss = sum(losses)/len(losses)
    # regularization
    tot_loss = loss + L2(model.params(), L2_lambda)
    # backward pass
    tot_loss.backward()
    # weights update
    for p in model.params():
        # the following learning rate is dynamic and decreases with each iteration
        p.data -= alpha * p.grad
    # accuracy
    directions = [(p.data>0) == (l>0) for p,l in zip(predictions, dataset_labels)]
    acc = (sum(directions)/len(directions))*100
    print(f'epoch:{epoch}, loss:{tot_loss.data}, accuracy:{round(acc,2)}%')
    epoch += 1
print("==> DONE")

# save trained model
print(f"==> Exporting model to JSON file...", end=' ')
model.to_json('./model/trained_model.json')
print("DONE")

if debug_mode:
    # use model on some value and generate the associated computational graph
    print('==> Using model on single datapoint to render the computational graph...', end=' ')
    inp = [list(map(Value, i)) for i in testing_data]
    pred = list(map(model, inp))
    computational_graph(pred[0], 'trained_model_graph')
    print('DONE')
