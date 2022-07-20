# these classes all uses the Value class underneath to compose the computational graph
# associated with the net in order to compute the gradients with the mutivariance chain rule
# while the params update with gradient descent is made in demo.py file
import random
import json
from capmangrad.engine import Value
from capmangrad.utils import ANSWER

class Base:
    def zero_grad(self):
        for p in self.params():
            p.grad = 0

    # this is a base class without any weights, biases or anything else
    def params(self):
        return []


class Neuron(Base):
    def __init__(self, inp_weights=0, nonlin=True):
        self.weights = [Value(random.uniform(-1,1), 'WEIGHT') for wi in range(inp_weights)]
        self.bias = Value(0, 'BIAS')
        self.nonlin = nonlin

    def __call__(self, inputs):
        # inp_data = inp_data if isinstance(Value, inp_data) else Value(inp_data)
        neuron_output = sum([xi*wi for xi, wi in zip(inputs, self.weights)], self.bias)
        neuron_output = neuron_output.relu() if self.nonlin else neuron_output
        return neuron_output

    def __repr__(self):
        return 'weights={},bias={},type={}'.format(self.weights, self.bias, 'ReLU' if self.nonlin else 'Linear')

    def to_json(self):
        return '{{"weights":{},"bias":{},"type":{}}}'.format(list(map(lambda w: w.data, self.weights)), self.bias.data, '"ReLU"' if self.nonlin else '"Linear"')

    def params(self):
        return self.weights + [self.bias]

    @staticmethod
    def from_json(w_list, b, n_type):
        # create a dummy neuron and fill it with the imported values
        n = Neuron()
        n.weights = [Value(n, 'WEIGHT') for n in w_list]
        n.bias = Value(b, 'BIAS')
        n.nonlin = True if n_type != 'Linear' else False
        return n


class Layer(Base):
    def __init__(self, inp_weights=0, neurons=0, **kwargs):
        self.neurons = [Neuron(inp_weights, **kwargs) for n in range(neurons)]

    def __call__(self, inputs):
        # each input in the list got multiplied by the respective weight in the neuron
        out = [n(inputs) for n in self.neurons]
        # return the only element if the layer is the output layer (which in this arch contains only 1 node)
        # return the result as a list instead (i.e. if it's a middle layer)
        # useful for map when doing predictions and losses
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        return 'layer={}'.format(self.neurons)

    def to_json(self):
        nl = [n.to_json() for n in self.neurons]
        neurons = ','.join(nl)
        return f'{{"layer":[{neurons}]}}'

    def params(self):
        return [p for neuron in self.neurons for p in neuron.params()]

    @staticmethod
    def from_json(json_data):
        # create a dummy layer and fill it with the imported neurons
        l = Layer()
        l.neurons = [Neuron.from_json(n['weights'], n['bias'], n['type']) for n in json_data['layer']]
        return l


class Model(Base):
    def __init__(self, inputs_number=0, network_arch=[], debug_mode=False):
        arch = [inputs_number] + network_arch
        # if in debug_mode always create a model with the same random weights
        # also useful in cross validation as it freezes the model while looking for the best hyperpar
        if debug_mode:
            random.seed(ANSWER)
        # nonlin used to prevent output layer nodes from being nonlinear
        self.layers = [
            Layer(arch[l], arch[l+1], nonlin=(l!=len(network_arch)-1))
            for l in range(len(network_arch))
        ]

    def __call__(self, inputs):
        # forward pass
        res = inputs
        for l in self.layers:
            res = l(res)
        # return list of outputs, one for each neuron in the last layer
        return res

    def __repr__(self):
        return 'model={}'.format(self.layers)

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def arch(self):
        return self.arch[0], self.arch[1:]

    def to_json(self, path=None):
        if not path:
            raise ValueError("path parameter to is missing")
        ll = [l.to_json() for l in self.layers]
        layers = ','.join(ll)
        model_json = f'{{"model":[{layers}]}}'
        with open(path, 'w') as json_file:
            json_file.write(model_json)

    @staticmethod
    def from_json(path=None):
        model_json = None
        if not path:
            raise ValueError("path parameter is missing")
        # read json file
        try:
            with open(path, 'r') as f:
                model_json = dict(json.load(f))
            m = Model()
            m.layers = [Layer.from_json(l) for l in model_json['model']]
            # print(model_dict)
        except FileNotFoundError:
            raise FileNotFoundError("Wrong file or file path")
        # create a dummy model and fill it with the imported layers
        return m
