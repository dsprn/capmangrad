import pytest
import torch
from capmangrad.nn import Model
from sklearn.datasets import make_moons
from capmangrad.xval import _group
from capmangrad.engine import Value
from capmangrad.utils import MSE, SVM_maxmargin, L2, data_ratio
    

# MODEL RELATED TESTS AND FIXTURES
@pytest.fixture
def model():
    # create a model with a known parameters' seed
    return Model(2, [16, 16, 1], True)

@pytest.fixture
def json_model():
    return Model.from_json('./tests/data/test_model.json')

@pytest.fixture
def input_data():
    dataset_vals, dataset_labels = make_moons(n_samples=5, noise=0.1)
    dataset_labels = dataset_labels*2 - 1
    return dataset_vals

@pytest.fixture
def path():
    return './tests/data/test_model.json'

def test_json(model, path):
    model.to_json(path)
    # creating new model from stored json
    json_model = Model.from_json(path)
    # test if repr of imported model is equal to repr of declared model
    assert repr(json_model) == repr(model)

def test_predictions(json_model, model, input_data):
    # test if the imported model get the same predictions as the runtime one (on the same inputs)
    json_pred = list(map(json_model, input_data)) 
    runtime_pred = list(map(model, input_data))
    assert len(json_pred) == len(runtime_pred)
    assert all([jp.data==rp.data for jp,rp in zip(json_pred,runtime_pred)])

def test_L2(model):
    reg = L2(model.params())
    # multiply the default L2 lambda value by the sum of the squares of the parameters in the debug model
    assert reg.data == 0.0001*106.49976797404548


# CROSS VALIDATION RELATED TESTS AND FIXTURES
@pytest.fixture
def dummy_data_list():
    return [9,3,2,51,7,18,93,32,11,47]

def test_data_ratio(dummy_data_list):
    training_data = [9,3,2,51,7,18,93,32]
    testing_data = [11,47]
    tr,te = data_ratio(dummy_data_list)
    assert len(tr) == len(training_data)
    assert len(te) == len(testing_data)
    assert all([x == y for x,y in zip(tr,training_data)])
    assert all([x == y for x,y in zip(te,testing_data)])

def test_group(dummy_data_list):
    groups = _group(dummy_data_list, k=5)
    assert all([len(g)==2 for g in groups])
    assert groups[0]==[9,3]
    assert groups[1]==[2,51]
    assert groups[2]==[7,18]
    assert groups[3]==[93,32]
    assert groups[4]==[11,47]


# LOSS FUNCTIONS RELATED TESTS AND FIXTURES
@pytest.fixture
def dummy_prediction():
    return 0.8

@pytest.fixture
def dummy_expectation():
    return 0.85

def test_SVM_maxmargin(dummy_prediction, dummy_expectation):
    err=SVM_maxmargin(dummy_prediction, dummy_expectation)
    assert round(err.data,4) == 0.32

def test_MSE(dummy_prediction, dummy_expectation):
    err=MSE(dummy_prediction, dummy_expectation)
    assert round(err.data,4) == 0.0025


# ENGINE RELATED TESTS
def test_ops():
    import torch

    x = Value(-42)
    z = 2*x+2+x
    q = z.relu()+z*x
    h = (z*z).relu()
    y = h+q+q*x
    y.backward()
    xcmg, ycmg = x, y

    x = torch.Tensor([-42]).double()
    x.requires_grad = True
    z = 2*x+2+x
    q = z.relu()+z*x
    h = (z*z).relu()
    y = h+q+q*x
    y.backward()
    xpt, ypt = x, y

    assert ycmg.data == ypt.data.item()
    assert xcmg.grad == xpt.grad.item()
