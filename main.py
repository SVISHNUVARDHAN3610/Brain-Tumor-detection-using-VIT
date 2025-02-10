import Transformer as ViT
import Dataset as DatasetLoader
import Trainer as Trainer

import torch
from torch import nn, optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import transforms
from torchinfo import summary

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"


vit = ViT.ViT(img_size=128,
          in_channels=3, 
          patch_size=16,
          num_transformer_layers=12, 
          embedding_dim=256,
          mlp_size=1024,
          num_heads=8,
          value_dim=64,
          key_dim=64,
          attn_dropout=0.1,
          mlp_dropout=0.1,
          embedding_dropout=0.1,
          num_classes=4)

summary(model=vit,
        input_size=(32, 3, 128, 128),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

#root_dir = "D:\\Research\\Image Classification\\Aerial Landscape Images Dataset" 
root_dir = "D:\Research\Image Classification\Brain Tumor Dataset"
batch_size = 32
image_size = (128, 128)

mean = torch.tensor([0.2104, 0.2104, 0.2104])
std = torch.tensor([0.1873, 0.1873, 0.1873])

train_transforms = transforms.Compose([
    transforms.Resize(image_size),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 15)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
    #transforms.RandomSolarize(threshold=192.0),
    transforms.RandomAdjustSharpness(sharpness_factor=1.2),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomEqualize(p=0.5),                            
    transforms.ToTensor(), 
    #transforms.Normalize(mean=[0.3779, 0.3927, 0.3443], std=[0.1356, 0.1234, 0.1182]) # Aerial Landscape Dataset
    transforms.Normalize(mean=[0.2104, 0.2104, 0.2104], std=[0.1873, 0.1873, 0.1873]) # Brain Tumor Dataset
])

test_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.2104, 0.2104, 0.2104], std=[0.1873, 0.1873, 0.1873])
])

loader = DatasetLoader.ImageFolderDatasetLoader(root_dir, batch_size=batch_size, transforms=None, seed=seed)
train_loader, test_loader = loader.get_train_test_loader(test_size=0.2, train_transforms=train_transforms, test_transforms=test_transforms)

def imshow(images, labels, num_images, names, mean, std):
    indices = torch.randperm(images.size(0))[:num_images]
    
    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_images, i + 1)
        
        img = images[idx].clone()
        img = img.mul_(std[:, None, None]).add_(mean[:, None, None])
        
        plt.imshow(img.permute(1, 2, 0) )
        plt.title(names[labels[idx].item()])
        plt.axis('off')

    plt.show()


num_images = 5
names = loader.get_label_to_name()
images, labels = next(iter(train_loader))

imshow(images, labels, num_images, names, mean, std)

def get_class_distribution(data_loader):
    return Counter([data_loader.dataset.dataset.samples[idx][1] for idx in data_loader.dataset.indices])

def get_sorted_class_names_and_frequencies(distribution, name_mapper):
    labels = list(distribution.keys())
    frequencies = list(distribution.values())
    
    class_names = [name_mapper[label] for label in labels]
    
    sorted_indices = sorted(range(len(class_names)), key=lambda i: class_names[i])
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_frequencies = [frequencies[i] for i in sorted_indices]
    
    return sorted_class_names, sorted_frequencies

def plot_class_distribution(class_names, frequencies):

    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, frequencies, color='skyblue')

    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


train_distribution = get_class_distribution(train_loader)
names = loader.get_label_to_name()
class_names, frequencies = get_sorted_class_names_and_frequencies(train_distribution, names)
plot_class_distribution(class_names, frequencies)


learning_rate = 0.00001
weight_decay = 0.01
loss_fn = nn.CrossEntropyLoss()
model_name = "model"

optimizer = optim.AdamW(vit.parameters(), lr=learning_rate, weight_decay=weight_decay)
trainer = Trainer.Trainer(vit, optimizer, loss_fn, model_name=model_name, device=device, patience=30)

trainer.train(train_loader, test_loader, epochs=500)

# Load the trained model of Vit on brain tumor classification
loaded_model = torch.load("brain_vit_128_209e_8837a.pt")

# Set the model to evaluation mode
loaded_model.eval()

# Function to display images with correct and predicted labels
def show_images(images, labels, predictions, probs, class_names):
    
    mean = torch.tensor([0.2104, 0.2104, 0.2104], device=images.device)  # Move mean to the same device as images
    std = torch.tensor([0.1873, 0.1873, 0.1873], device=images.device)   # Move std to the same device as images

    # Apply denormalization using broadcasting
    images = images * std[None, :, None, None] + mean[None, :, None, None]
    
    plt.figure(figsize=(12, 6))  # Set a suitable figure size
    for i in range(10):  # Display only the first 10 images
        plt.subplot(2, 5, i + 1)  # Two rows, five columns
        image = images[i].permute(1, 2, 0).cpu().numpy()  # Permute to put channel last
        plt.imshow(image)  # Show the image
        plt.axis('off')  # Remove axes

        # Title with true and predicted labels along with probability
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predictions[i].item()]
        prob = probs[i].item()
        plt.title(f"T: {true_label}\nP: {pred_label} ({prob:.2f})")

    plt.tight_layout()
    plt.show()

    # Extract the first batch
inputs, labels = next(iter(test_loader))

# Perform inference (disable gradient computation for faster inference)
with torch.no_grad():
    outputs = loaded_model(inputs)  # Forward pass through the model

# Get predicted labels and probabilities
probs = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
predictions = torch.argmax(probs, dim=1)  # Get the predicted class

# Assuming 'class_names' is a list of class names corresponding to the labels
show_images(inputs, labels, predictions, torch.max(probs, dim=1).values, class_names)