
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms,models
from collections import OrderedDict
import json
import argparse
from PIL import Image,ImageOps 

parser = argparse.ArgumentParser()

parser.add_argument("-e","--epochs",help = "No of Epochs", metavar='', type = int)
parser.add_argument("-l","--learning_rate",help = "learning rate", metavar='',type = float )
parser.add_argument("-d","--device", help = "device is cuds or cpu", metavar='',type = str )
parser.add_argument("-m","--model_type", help = "model type to train", metavar='',type = str )
parser.add_argument("-hl","--hidden_layer", help = "first hidden layer", metavar='', type = int )

args = parser.parse_args()

if args.epochs:
    epochs = args.epochs
    print("epochs given as argument : {} ".format(epochs))
else :
    print("No epochs given as argument : ")
    # set to 70 for better accuracy
    epochs = 1

if args.learning_rate:
    learning_rate = args.learning_rate
    print("Learning_rate as argument : {} ".format(learning_rate))
else :
    print("No learning_rate as an argument: default is 0.001")
    learning_rate = 0.001

if args.device:
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else :
        device = "cpu"
    print("Device as argument : {} ".format(device))
else :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("No device as an argument: default is {} ".format(device))

if args.model_type:
    network = args.model_type
    if network == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif network == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif network == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif network == 'densenet121':
        model = models.densenet121(pretrained=True)
    else :
        model = models.vgg16(pretrained=True)
        print("Choosing default model to train : {}".format(network))
        print ("\t U can choose below models: \n \t vgg16 \t alexnet \t densenet161 \t densenet121 ")
else :
    network = 'vgg16'
    model = models.vgg16(pretrained=True)
    print("Choosing default model to train : {}".format(network))
    print ("U can choose below models \n vgg16 \t alexnet \t densenet161 \t densenet121")
    
if args.hidden_layer:
    hidden_layer =  args.hidden_layer
    print("Hidden layer as input arg : {} ".format(hidden_layer))
else :
    hidden_layer = 4096
    print("Hidden layer as default : {} ".format(hidden_layer))
    
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) ])

valid_data_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ])

test_data_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
print("Loading Image Data")
train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_data_transforms)
test_image_datasets = datasets.ImageFolder(test_dir, transform=test_data_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
testloader  = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f) 

# Build and train your network
print("Building Network")
#model = models.vgg16(pretrained=True)
print(" Model to training is  : {}".format(network))

networks = {"vgg16" : 25088,
           "alexnet" : 9216,
           "densenet161" : 2208,
           "densenet121" : 1024}

for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(networks[network], hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_layer, 2048)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(p=0.2)),
                          ('fc3',nn.Linear(2048, 102)), 
                          ('output', nn.LogSoftmax(dim=1))        
                          ]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(),lr=learning_rate)

print("Networking Training started")
steps = 0

#Finding Device
model.to(device)

train_losses, valid_losses = [], []
for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() 
        running_loss += loss.item()
    else:
        valid_loss = 0
        accuracy = 0
        #print("Validation started : ")
        # Turn off gradients for validation, saves memory and computations
        model.eval()
        with torch.no_grad(): 
            for images, labels in validloader:
                inputs, labels = images.to(device), labels.to(device)
                log_ps = model.forward(inputs)
                valid_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        model.train()
        train_losses.append(running_loss/len(trainloader))
        valid_losses.append(valid_loss/len(validloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))  
        
# Do validation on the test set
print("Testing started ")
test_loss = 0
accuracy = 0
for images, labels in testloader:
    #print(images.shape)
    inputs, labels = images.to(device), labels.to(device)
    log_ps = model.forward(inputs)
    #test_loss += criterion(log_ps, labels)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor))
      
print("Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

# Save the checkpoint
print("Saving Network state")
checkpoint = {'input_size' : networks[network],
              'output_size' : 102,
              'model_type' : network,
              'hidden_layer' : hidden_layer,
              'epochs' : epochs,
              'lr' : learning_rate,
              'optimizer' : optimizer.state_dict(),
              'class_to_idx' : cat_to_name,
              'state_dict' : model.state_dict()}

torch.save(checkpoint,'checkpoint.pth')
