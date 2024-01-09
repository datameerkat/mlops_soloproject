import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model import MyAwesomeModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    lr = 1e-4
    batch_size = 256
    num_epochs = 10
    train(lr, batch_size, num_epochs)
    model_checkpoint = "model.pt"
    evaluate(model_checkpoint, batch_size)

def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(batch_size)

    # Training loop
    model = MyAwesomeModel()
    train_dataloader, _ = create_dataloader(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss}")

    torch.save(model, "model.pt")

def evaluate(model_checkpoint, batch_size):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    _, test_dataloader = create_dataloader(batch_size)
    
    model = torch.load(model_checkpoint)
    model.eval()

    test_preds = [ ]
    test_labels = [ ]
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            test_preds.append(y_pred.argmax(dim=1).cpu())
            test_labels.append(y.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    print(f"Accuracy: {(test_preds == test_labels).float().mean()} %")

def create_dataloader(batch_size):
    #processed_dir = '/home/datameerkat/MLOPs/Course_material/mlops_soloproject/data/processed'
    script_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(script_dir, '..', 'data/processed')
    train_tensor_file = 'train_tensor.pt'
    train_labels_file = 'train_labels.pt'
    test_tensor_file = 'test_tensor.pt'
    test_labels_file = 'test_labels.pt'

    train_data = torch.load(os.path.join(processed_dir, train_tensor_file))
    train_labels = torch.load(os.path.join(processed_dir, train_labels_file))
    test_data = torch.load(os.path.join(processed_dir, test_tensor_file))
    test_labels = torch.load(os.path.join(processed_dir, test_labels_file))

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader



if __name__ == "__main__":
    main()