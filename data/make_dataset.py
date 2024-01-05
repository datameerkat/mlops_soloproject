import torch
import os

def load_data(raw_dir):
    """Load raw MNIST data."""
    train_data, train_labels = [], []
    for i in range(5):
        train_data.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_labels.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0).unsqueeze(1).float()
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load(f"{raw_dir}/test_images.pt").unsqueeze(1).float()
    test_labels = torch.load(f"{raw_dir}/test_target.pt")

    return train_data, train_labels, test_data, test_labels

def normalize_data(tensor):
    """Normalize tensor to have mean 0 and std 1."""
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

def save_data(tensor, labels, filename, path):
    """Save processed tensors to the given path."""
    torch.save(tensor, os.path.join(path, f'{filename}_tensor.pt'))
    torch.save(labels, os.path.join(path, f'{filename}_labels.pt'))

def main():
    raw_dir = '/home/datameerkat/MLOPs/Course_material/mlops_soloproject/data/raw/corruptmnist'
    processed_dir = '/home/datameerkat/MLOPs/Course_material/mlops_soloproject/data/processed'

    # Load raw data
    train_data, train_labels, test_data, test_labels = load_data(raw_dir)

    # Normalize data
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    # Save processed data
    save_data(train_data, train_labels, 'train', processed_dir)
    save_data(test_data, test_labels, 'test', processed_dir)

    print("Data processing complete. Processed data saved to:", processed_dir)

if __name__ == "__main__":
    main()
