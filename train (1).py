import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image


import argparse

parser = argparse.ArgumentParser (description = 'Parser _ training script')

parser.add_argument ('data_dir', help = 'Input data directory. Mandatory', type=str)
parser.add_argument ('--save_dir', help = 'Input saving directory. Optional',  type=str)
parser.add_argument ('--arch', help = 'Default is Alexnet, otherwise input VGG13', type=str, default='alexnet')
parser.add_argument ('--learning_r', help = 'Learning rate - default is 0.003', type = float, default = 0.003)
parser.add_argument ('--hidden_units', help = 'Hidden units. Default val 2048', type = int, default = 2048)
parser.add_argument ('--epochs', help = 'Epochs as integer - default is 1', type = int, default = 1)
parser.add_argument ('--GPU', help = "Input GPU if you want to use it", type = str)

args = parser.parse_args()

data_dir = 'args.data_dir'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                     ])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_data , batch_size=64, shuffle=True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    model = torchvision.models.vgg16(pretrained=True)

for x in model.parameters():
        x.requires_grad = False
        
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))        
model.classifier = classifier
model=model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), args.learning_r)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = args.epochs
print_every = 25
steps = 0

for e in range(epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(steps)
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():#
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    output = model.forward(inputs)
                    loss = criterion(output, labels)
                    test_loss += loss.item()

                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
# TODO: Save the checkpoint 

model.class_to_idx = train_data.class_to_idx
checkpoint = {'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'learning rate': 0.003,
              
             }
torch.save(checkpoint, 'checkpoint.pth')         

if __name__ == "__main__":
    main()