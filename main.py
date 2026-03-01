import dataset
import model
import training
import numpy as np
import torch
from torch.utils.data import DataLoader



if __name__ == '__main__':
    TARGET_FPS = 16
    train_dataset = dataset.VideoDataset(root_dir = 'dataset/train', label_to_idx_path = 'dataset/label_mapping.pkl', 
                                        target_fps = TARGET_FPS, training = True)

    valid_dataset = dataset.VideoDataset(root_dir = 'dataset/train', label_to_idx_path = 'dataset/label_mapping.pkl',
                                        target_fps = TARGET_FPS, training = False)

    # Split
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    indices = list(range(len(train_dataset)))

    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

    balanced_sampler = dataset.create_balanced_sampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size = 16, sampler = balanced_sampler, collate_fn = dataset.collate_fn, num_workers = 4)
    valid_loader = DataLoader(valid_dataset, batch_size = 16, shuffle = False, collate_fn = dataset.collate_fn, num_workers = 4)

    model = model.VitTransformer(num_classes = 100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = training.train_model(model, train_loader, valid_loader, num_epochs = 20, learning_rate = 1e-4, device = device)

    training.evaluate(model, 'dataset/public_test', 'dataset/label_mapping.pkl', 'public_test.csv', device, 'best_model.pth', TARGET_FPS)
    training.evaluate(model, 'dataset/private_test', 'dataset/label_mapping.pkl', 'prediction_test.csv', device, 'best_model.pth', TARGET_FPS)