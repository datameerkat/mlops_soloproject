from tests import _PROJECT_ROOT
import torch
from models import model

model = model.MyAwesomeModel()

def test_model():
    input = torch.ones(1 ,1, 28, 28)
    output = model(input)
    assert output.shape == (1, 10)