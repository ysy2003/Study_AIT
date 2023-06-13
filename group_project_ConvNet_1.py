import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split, DataLoader
import matplotlib.patches as patches

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 30 # 30epoch的话，准确率会到97%左右但是耗时太久了。15epoch的话准确率稍低一些（为了降低每次跑的时间我用了3）
batch_size = 10
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),  # Data augmentation: random horizontal flip
transforms.RandomRotation(10),  # Data augmentation: random rotation by 10 degrees 自己加的，因为有些图片是倾斜的
    transforms.RandomCrop(32, padding=4),  # Data augmentation: random crop with padding of 4 pixels 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Data augmentation: color transformation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

# Load your dataset
dataset = torchvision.datasets.ImageFolder(root='C://Users//25435//Desktop//deeplearning//dataset', transform=transform)
# Update the root path to match your dataset location

# Split your dataset into train, validation and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('cat', 'goose','bird')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []

n_total_steps = len(train_loader)
# Training loop
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        l = loss(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.item()

        if (i + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {l.item():.4f}')

    else:
        # Validation after each epoch
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                l = loss(outputs, labels)
                val_loss += l.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Validation Accuracy of the model on the {total} validation images: {100 * correct / total}%')

        model.train()
    # save the average train loss for this epoch
    train_losses.append(train_loss / len(train_loader))
    # save the average validation loss for this epoch
    val_losses.append(val_loss / len(val_loader))

# Save the trained model
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
print('Finished Training')

# Plot the training and validation loss 把loss可视化了
plt.figure(figsize=(12, 4))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss values')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Function to draw rectangles on the images with smaller size 红色矩形框是cat，绿色矩形框是bird，蓝色矩形框是goose，增加准确度
def draw_rectangles(images, labels, predictions):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    for i, ax in enumerate(axes):
        # Unnormalize the image
        image = images[i] / 2 + 0.5
        npimg = image.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))

        # Get the predicted label and true label
        predicted_label = classes[predictions[i]]
        true_label = classes[labels[i]]

        # Calculate the rectangle size
        rect_width = 16
        rect_height = 16

        # Calculate the top-left corner position of the rectangle
        rect_x = (32 - rect_width) / 2
        rect_y = (32 - rect_height) / 2

        # Draw a rectangle around the predicted object
        if predicted_label == 'bird':
            rect_color = 'g'  # green for bird
        elif predicted_label == 'cat':
            rect_color = 'r'  # red for cat
        else:
            rect_color = 'b'  # blue for goose

        rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1, edgecolor=rect_color, facecolor='none')
        ax.add_patch(rect)

        # Set the title of the image with predicted and true labels
        ax.set_title(f'Predicted: {predicted_label}\nTrue: {true_label}')
        ax.axis('off')

    plt.show()


# Evaluate the trained model on the test set
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(3)]
    n_class_samples = [0 for i in range(3)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

        # Draw rectangles on the images with smaller size
        draw_rectangles(images, labels, predicted)

    model.train()  # Set the model back to training mode

    # Calculate accuracy metrics
    accuracy = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {accuracy:.2f} %')

    for i in range(3):
        class_accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {class_accuracy:.2f} %')


