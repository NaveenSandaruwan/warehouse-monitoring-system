import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import os

# Step 1: Define the Dataset Class
class WarehouseActionDataset(Dataset):
    def __init__(self, video_dir, labels, transform=None, seq_length=16):
        self.video_dir = video_dir
        self.labels = labels
        self.transform = transform
        self.seq_length = seq_length
        self.video_files = os.listdir(video_dir)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        frames = self._extract_frames(video_path)
        label = self.labels[idx]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)  # Stack frames into a tensor
        return frames, label

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)
            if len(frames) == self.seq_length:
                break
        cap.release()
        return frames

# Step 2: Define the 3D CNN Model
class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.fc1 = nn.Linear(128 * 8 * 8 * 8, 512)  # Adjust based on input size
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Step 3: Define Transformations and DataLoader
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # Resize frames
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Example labels (replace with your actual labels)
video_dir = "path/to/your/videos"
labels = [0, 1, 0, 1, 2]  # 0: Picking, 1: Packing, 2: Walking
dataset = WarehouseActionDataset(video_dir, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Step 4: Train the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActionRecognitionModel(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.permute(0, 2, 1, 3, 4))  # Permute for 3D CNN input
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}], Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

# Step 5: Save the Model
torch.save(model.state_dict(), "warehouse_action_recognition_model.pth")

# Step 6: Inference
def predict_action(video_path, model, transform, seq_length=16):
    model.eval()
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)
        if len(frames) == seq_length:
            break
    cap.release()
    frames = torch.stack(frames).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(frames.permute(0, 2, 1, 3, 4))
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Example usage
video_path = "path/to/test_video.mp4"
predicted_action = predict_action(video_path, model, transform)
print(f"Predicted Action: {predicted_action}")