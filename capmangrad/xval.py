# RESOURCES HUNGRY, NAIVE IMPLEMENTATION
# in this implementation of cross validation, xval objects could eat a lot of memory and cpu so they put
# a cap on what you can do, hence the name of the project (and the following funny ascii art comment)

        #######
     #############
   #######   #######
  ########   #####
 ###############
##############       C      R      C
############        CPU    RAM    CPU
##############       U      M      U
 ###############
  ################
   #################
     #############
        #######


import sys
import numpy as np
from capmangrad.nn import Neuron, Layer, Model
from capmangrad.engine import Value
from capmangrad.utils import L2


class xval:
    """
    cross validation to get the best hyperparameters (in this case L2 penalty value)
    """
    def __init__(self, data, labels, model_arch, hyper_range, alpha=0.001, loss_fn=None, debug_mode=False, k=10):
        self.model = None
        if not isinstance(model_arch, list):
            raise TypeError("Model architecture should be in the form: [inputs, [hidden layer, ..., hidden layer, output]]")
        self.model_arch = model_arch
        self.k = k
        self.alpha = alpha
        if not loss_fn:
            raise TypeError("None is not a valid loss_fn")
        self.loss_fn = loss_fn
        # data
        self.values = _group(data, self.k)
        self.labels = _group(labels, self.k)
        # hyperparameter range
        # self.lambda_range = np.arange(0, 0.1, 0.001)
        self.hyper_range = hyper_range
        self.cv_scores = {}
        self.debug_mode = debug_mode

    def __call__(self):
        print(f'==> Using Cross Validation to look for best hyperparameter in values ranging from {round(min(self.hyper_range),5)} to {round(max(self.hyper_range),5)}')
        # for each hyper parameter value
        for h in self.hyper_range:
            losses = []
            scores = []
            # pick a testing group and a testing label
            # train the model on the other groups and test it against the selected one
            # then loop to change the combination
            for ki in range(self.k):
                # copying lists here because pop modifies the state of its source
                training_values = self.values.copy()
                training_labels = self.labels.copy()
                holdout_values = training_values.pop(ki)
                holdout_labels = training_labels.pop(ki)
                self._mini_train(training_values, training_labels, h)
                # scores.append(self._holdout_test(holdout_values, holdout_labels))
                ls, ac = self._holdout_test(holdout_values, holdout_labels)
                losses.append(ls)
                scores.append(ac)
            # averaging scores for this hyper parameter and saving the result
            loss = sum(losses)/len(losses)
            avg_score = sum(scores)/len(scores)
            # self.cv_scores[loss] = h
            if avg_score not in self.cv_scores:
                self.cv_scores[avg_score] = [(loss.data, h)]
            else:
                self.cv_scores[avg_score].append((loss.data, h))
            if self.debug_mode:
                print(f'hyperparameter:{round(h,5)}, loss:{loss.data}, accuracy:{round(avg_score,2)}%')
            # break out of the loop when accuracy starts dropping
            # i.e. the best hyperparameter has already been found
            # if avg_score<max(self.cv_scores.keys())
            #     break
        # return the hyper parameter value associated with the minimum loss (because used for L2 in this project)
        # hyperpar = self.cv_scores[min(self.cv_scores.keys())]
        # return the hyper parameter value associated with the maximum accuracy
        scores_list = self.cv_scores[max(self.cv_scores.keys())]
        # get the hyperpar from the only tuple present in the list
        if len(scores_list) == 1:
            hyperpar = scores_list[0][1]
        # get the hyperpar from the tuple with the lowest loss in the list
        else:
            dummy_loss = sys.maxsize
            for el in scores_list:
                if el[0]<dummy_loss:
                    dummy_loss,hyperpar = el[0],el[1]
        # hyperpar = self.cv_scores[max(self.cv_scores.keys())]
        if self.debug_mode:
            # print(f'==> Cross Validation computed scores:{self.cv_scores}')
            print(f'==> Best hyperparameter in given range: {round(hyperpar,4)}')
        return hyperpar

    def _mini_train(self, values, labels, hyperpar_value):
        # xval always gets a Model with the same random parameters to keep it fixed while looping through hyperpar values
        self.model = Model(self.model_arch[0], self.model_arch[1:], True)
        # check the model has always the same weights (for development purposes)
        # print(f"debug_mode:{self.debug_mode}, model_parameters_sum:{sum(self.model.params()).data}")
        # train the model once on each one of the training groups
        for group, expectations in zip(values, labels):
            # convert data to input values
            inputs = [list(map(Value, i)) for i in group]
            # train for 10 times on the same inputs group
            for epoch in range(10):
                # prepping for new forward pass
                self.model.zero_grad()
                # making predictions
                preds = list(map(self.model, inputs))
                # computing loss with regularization
                losses = list(map(self.loss_fn, preds, expectations))
                loss = sum(losses)/len(losses)
                tot_loss = loss + L2(self.model.params(), hyperpar_value)
                # backward pass and weights update
                tot_loss.backward()
                for p in self.model.params():
                    p.data -= self.alpha * p.grad

    def _holdout_test(self, values, labels):
        # test the model on the remaining group and return the loss
        inputs = [list(map(Value, i)) for i in values]
        preds = list(map(self.model, inputs))
        losses = list(map(self.loss_fn, preds, labels))
        loss = sum(losses)/len(losses)
        # accuracy
        directions = [(p.data>0) == (l>0) for p,l in zip(preds, labels)]
        acc = (sum(directions)/len(directions))*100
        # print(f'==> TESTED ON HOLDOUT (loss:{loss.data}, accuracy:{acc}%)')
        return loss, acc

    def __repr__(self):
        return f"(data:{self.values}, labels:{self.labels}, groups lenght:{self.k}, cross validation scores:{self.cv_scores})"


def _group(data, k=0):
    """
    split data into equal sized k groups
    """
    # if k=0 then use "Leave One Out Cross Validation"
    k = len(data) if k == 0 else k
    # number of elements for each group
    size = int(len(data) / k)
    groups = []
    start = 0
    for i in range(k):
        groups.append(data[start:start+size])
        start += size
    return groups
