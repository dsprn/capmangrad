# autodiff engine (using the mutivariance chain rule to compute gradients w.r.t some value)
class Value:
    def __init__(self, data, type='INPUT', _children=(), _op=''):
        self.data = data
        self.grad = 0           # at each node it contains the derivative of self w.r.t. the value on which backward was called
        self.type = type
        self._prev = set(_children)
        self._backward = lambda: None   # for backpropagation (each operation has its own)
        self._op = _op                  # for debugging purposes with graphviz

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, 'ADD_RESULT', (self, other), '+')
        def _backward():
            # when _backward is called self and other are the inputs to the operation
            # multiplication is part of the (mutivariance version of the) chain rule
            # addition is part of the multivariance version of the chain rule (to sum different path)
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, 'MUL_RESULT', (self, other), '*')
        def _backward():
            # when _backward is called self and other are the inputs to the operation
            # multiplication is part of the (mutivariance version of the) chain rule
            # addition is part of the multivariance version of the chain rule (to sum different path)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __lt__(self, other):
        return self.data < other.data

    def __gt__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data > other.data
    
    def __ge__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data >= other.data

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __rsub__(self, other):
        return self - other

    def __truediv__(self, other):
        # used when computing mean of losses in demo.py file before L2 regularization
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __repr__(self):
        return 'Value({},{},{})'.format(self.type, self.data, self.grad)
        # return '{}'.format(self.data)

    def __pow__(self, exponent):
        # used in loss function
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, 'POWER_RESULT', (self,), f'**{exponent}')
        def _backward():
            # when _backward is called self and exponent are the inputs to the operation
            # multiplication is part of the (mutivariance version of the) chain rule
            # addition is part of the multivariance version of the chain rule (to sum different path)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value((self.data > 0) * self.data, 'RELU_RESULT', (self,), 'ReLU')
        def _backward():
            # multiplication is part of the (mutivariance version of the) chain rule
            # addition is part of the multivariance version of the chain rule (to sum different path)
            self.grad += (0 if self.data < 0 else 1) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # a derivative of something w.r.t. to itself is 1
        # GRADIENT DESCENT cares about reducing the gradient of a param w.r.t the LOSS
        # so it computes the derivatice of parameter of the net w.r.t. the loss
        # and the first step is to set the derivative of the LOSS w.r.t. itself to 1
        # if this method is invoked on a value other than the loss it will start to compute gradients from that value
        self.grad = 1
        # order the NN computational graph with a topological sort
        topological_order = []
        visited = set()
        def topological_sort(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    topological_sort(child)
                topological_order.append(node)
        topological_sort(self)
        for node in reversed(topological_order):
            node._backward()
