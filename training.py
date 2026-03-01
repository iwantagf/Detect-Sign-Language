from model import VitTransformer
from dataset import VideoDataset
from dataset import create_balanced_sampler
from dataset import read_video
from dataset import collate_fn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os
from tqdm import tqdm
import csv
from sklearn.metrics import precision_recall_fscore_support


def evaluate(model, folder_path, label_to_idx_path, output_csv="prediction.csv",
             device="cuda" if torch.cuda.is_available() else "cpu", model_path=None, target_fps=16):
    # Load model
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    
    model = model.to(device)
    model.eval()

    with open(label_to_idx_path, 'rb') as f:
        label_mapping = pickle.load(f)
    
    idx_to_label = {v: k for k, v in label_mapping.items()}
    
    video_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))])

    predictions = []

    dataset = VideoDataset(root_dir=folder_path, label_to_idx_path=label_to_idx_path, target_fps=target_fps, training=False)

    with torch.no_grad():
        for video_file in tqdm(video_files, desc="Evaluating"):
            video_path = os.path.join(folder_path, video_file)

            try:
                frames = read_video(video_path)
                frames = dataset._downsample_frames(frames)
                frames = dataset._normalize(frames)
                frames = frames.unsqueeze(0).to(device) # (1, T, C, H, W)

                outputs = model(frames)

                _, predicted = outputs.max(1)
                label_idx = predicted.item()
                label_name = idx_to_label[label_idx]

                predictions.append((video_file, label_name))
            except Exception as e:
                print(f"Error {video_file}: {e}")

    with open(output_csv, mode = 'w', newline = '', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'label'])
        writer.writerows(predictions)
    
    print(f"Prediction saved to {output_csv}")
    print(f"Total video processed: {len(predictions)}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc = "Training")

    for batch in progress:
        frames, labels = batch['frames'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, preds, labels_all = 0, [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc = "Validating"):
            frames, labels = batch['frames'].to(device), batch['labels'].to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds.extend(outputs.max(1)[1].cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds, average = 'macro', zero_division = 0)
    
    return total_loss / len(dataloader), {'precision': precision * 100, 'recall': recall * 100, 'f1': f1 * 100}

def train_model(model, train_loader, valid_loader, num_epochs = 20, learning_rate = 1e-4, device = 'cuda'):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 3)
     
    best_f1 = 0
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, metrics = validate(model, valid_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Metrics: {metrics}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved")
        
        scheduler.step(valid_loss)
    
    return model

