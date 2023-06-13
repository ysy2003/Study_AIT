import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import copy
import time
import shutil
from copy import deepcopy

src_dataset_dir = 'C://Users//25435//Desktop//deeplearning//dataset' #跟ConvNet的一样
train_dataset_dir = 'C://Users//25435//Desktop//deeplearning//dataset_1/train' # 用一个新的文件夹名字来存储训练集
val_dataset_dir = 'C://Users//25435//Desktop//deeplearning//dataset_1/val' # 用一个新的文件夹名字来存储验证集（跟上面的一样）


class_names = ['cat', 'goose', 'bird']

train_ratio = 0.7

for class_name in class_names:
    os.makedirs(os.path.join(train_dataset_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dataset_dir, class_name), exist_ok=True)

    src_dir = os.path.join(src_dataset_dir, class_name)
    images = os.listdir(src_dir)
    
    np.random.shuffle(images)
    
    train_idx = int(len(images) * train_ratio)
    train_images = images[:train_idx]
    val_images = images[train_idx:]
    
    for image in train_images:
        shutil.copy(os.path.join(src_dir, image), os.path.join(train_dataset_dir, class_name, image))

    for image in val_images:
        shutil.copy(os.path.join(src_dir, image), os.path.join(val_dataset_dir, class_name, image))

# Define your dataset directory
data_dir = 'C://Users//25435//Desktop//deeplearning//dataset_1' #用新的文件夹名字

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=6, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)

def imshow(inp, title):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25,early_epochs=25,early_stopping=True,patience = 7): # num_epochs=25

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()

    train_losses = []
    val_losses = []

    # early stopping parameters initialization
    epochs_no_improve = 0
    early_stop = False
    min_val_loss = np.Inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # add loss to the corresponding list
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            # 如果这是验证阶段并且我们有一个新的最低验证损失，那么复制模型
            if phase == 'val' and epoch_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = epoch_loss
                best_model_wts = deepcopy(model.state_dict())
                best_acc = epoch_acc

            # 如果这是验证阶段并且损失没有减少，那么增加计数器
            elif phase == 'val' and epoch_loss >= min_val_loss:
                epochs_no_improve += 1
                # 如果损失没有减少超过patience个epoch，那么停止训练
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    early_stop = True
                    break
            else:
                continue

            break
        if early_stop:
            print("Stopped")
            break

        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses,label='Training loss')
    plt.plot(val_losses,label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return model


#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.

model = models.resnet18(pretrained=True)
#model = models.resnet18(weights='imagenet')

num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 3.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 3)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001)

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=10) # num_epochs=10,3改10的


#### ConvNet as fixed feature extractor ####
# Here, we need to freeze all the network except the final layer.
# We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
model_conv = torchvision.models.resnet18(pretrained=True)
#model_conv = torchvision.models.resnet18(weights='imagenet')

for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10) # 3改10的

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=3)