from tests import _PROJECT_ROOT
from models import model, train_model

N_train = 25000
N_test = 5000

def test_dataloader():
    train_loader, test_loader = train_model.create_dataloader(batch_size=2)
    assert len(train_loader.dataset) == N_train
    assert len(test_loader.dataset) == N_test