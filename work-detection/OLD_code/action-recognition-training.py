
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import cv2
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import json

class VideoDataset(Dataset):
    def __init__(self, root_dir: str, sequence_length: int = 16, transform=None):
        """
        Args:
            root_dir (str): Directory with all the video folders
                Structure should be:
                root_dir/
                    action1/
                        video1/
                            frame1.jpg
                            frame2.jpg
                            ...
                        video2/
                            ...
                    action2/
                        ...
            sequence_length (int): Number of frames per sequence
            transform: Optional transform to be applied on frames
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Create action to index mapping
        self.actions = sorted(os.listdir(root_dir))
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        
        # Save mapping to file
        with open('action_mapping.json', 'w') as f:
            json.dump({
                'action_to_idx': self.action_to_idx,
                'idx_to_action': {str(v): k for k, v in self.action_to_idx.items()}
            }, f)
        
        # Get all video sequences
        self.sequences = []
        for action in self.actions:
            action_path = os.path.join(root_dir, action)
            for video_folder in os.listdir(action_path):
                video_path = os.path.join(action_path, video_folder)
                frames = sorted(os.listdir(video_path))
                
                # Create sequences of consecutive frames
                for i in range(0, len(frames) - sequence_length + 1, sequence_length):
                    sequence_frames = frames[i:i + sequence_length]
                    if len(sequence_frames) == sequence_length:
                        self.sequences.append({
                            'frames': [os.path.join(video_path, f) for f in sequence_frames],
                            'action': action
                        })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        frames = []
        
        for frame_path in sequence['frames']:
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        # Stack frames into tensor
        frames = torch.stack(frames)
        label = self.action_to_idx[sequence['action']]
        
        return frames, label
'''
class VideoDataset(Dataset):
    def __init__(self, root_dir: str, sequence_length: int = 16, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        self.actions = sorted(os.listdir(root_dir))
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}

        with open('action_mapping.json', 'w') as f:
            json.dump({
                'action_to_idx': self.action_to_idx,
                'idx_to_action': {str(v): k for k, v in self.action_to_idx.items()}
            }, f)

        self.sequences = []
        for action in self.actions:
            action_path = os.path.join(root_dir, action)
            for video_file in os.listdir(action_path):
                video_path = os.path.join(action_path, video_file)
                if os.path.isfile(video_path) and video_path.endswith(('.mp4', '.avi', '.mov')):
                    video_frames = self.extract_frames(video_path)
                    if len(video_frames) >= self.sequence_length:
                        for i in range(0, len(video_frames) - self.sequence_length + 1, self.sequence_length):
                            sequence_frames = video_frames[i:i + self.sequence_length]
                            self.sequences.append({
                                'frames': sequence_frames,
                                'action': action
                            })

    def extract_frames(self, video_path: str) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        frames = []
        for frame in sequence['frames']:
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)
        label = self.action_to_idx[sequence['action']]
        return frames, label
'''

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ActionRecognitionModel, self).__init__()
        self.base_model = r3d_18(pretrained=pretrained)
        
        # Modify the final layer to match our number of classes
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def train_model(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    sequence_length: int = 16
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                           std=[0.22803, 0.22145, 0.216989])
    ])

    # Create dataset and dataloader
    dataset = VideoDataset(data_dir, sequence_length, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create model
    num_classes = len(dataset.actions)
    model = ActionRecognitionModel(num_classes=num_classes)
    model = model.to(device)

    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (frames, labels) in enumerate(dataloader):
            # Move data to device
            frames = frames.permute(0, 2, 1, 3, 4).to(device)  # [B, C, T, H, W]
            labels = labels.to(device)

            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')

        # Learning rate scheduling
        scheduler.step(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'accuracy': accuracy
            }, os.path.join(output_dir, 'best_model.pth'))

    print("Training completed!")

if __name__ == "__main__":
    # Example usage
    train_model(
        data_dir=r"C:\Users\anura\Downloads\OLD_code\Training_dir",
        output_dir=r"C:\Users\anura\Downloads\OLD_code\Training_dir",
        num_epochs=30,
        batch_size=8,
        learning_rate=0.001,
        sequence_length=16
    )
