import json
import math
import numpy as np
from graphviz import Digraph, Source
from capmangrad.engine import Value


# this is the answer to the ultimate question of life, the universe, and everything 
ANSWER = 42


class NoForwardPassYet(Exception):
    def __init__(self, message):
        super().__init__(message)


def _trace(root):
    # recursively build the sets of nodes and edges backwards (using _prev Value's field)
    # to take a look at the NN computational graph use the prediction as root node
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def _draw_dot(output_node, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = _trace(output_node)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    for n in nodes:
        dot.node(name=str(id(n)), 
                 label = "{ %s | data %.4f | grad %.4f }" % (n.type, n.data, n.grad), shape='record')
        # if there is an op add it to the graph
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


def computational_graph(prediction_node, file_name):
    """
    saves NN's associated computational graph to file_name after the last backward pass
    render the output file with graphviz
    """
    # TODO: only works on single-output NN, implement for multi-output NN too
    if prediction_node:
        # get a top to bottom graph instead of the default (ledt to right) one
        dot = _draw_dot(prediction_node, rankdir='TB')
        # get a verbatim DOT source code string to be rendered
        s = Source(str(dot))
        s.render(f'./computational_graph/{file_name}', format='svg', view=False)
    else:
        raise NoForwardPassYet('Do at least one Forward Pass to create the computational graph associated with this Neural Network')


def SVM_maxmargin(predicted, expected):
    expected = expected if isinstance(expected, Value) else Value(expected)
    return (1 + -expected*predicted).relu()


def L2(model_parameters, l=1e-4):
    """
    useful to prevent overfitting on the training data for networks with many parameters and high expressivity
    """
    # Note. LAMBDA is a positive value which can range from 0 to positive infinity, but typically is chosen between 0 and 10
    # reg = (10**l) * sum(p**2 for p in model_parameters)
    reg = l * sum(p**2 for p in model_parameters)
    return reg


# TODO: make MSE loss function work over batches (n>1)
# loss function (MSE where n=1)
def MSE(predicted, expected):
    '''
    it's the average of the squared difference between the original values and the predicted ones

    it's easy to compute its gradient
    '''
    expected = expected if isinstance(expected, Value) else Value(expected)
    return (predicted - expected) ** 2


def data_ratio(data, perc=0.8):
    till = int(len(data) * perc)
    training_data = data[:till]
    testing_data = data[till:]
    return training_data, testing_data
