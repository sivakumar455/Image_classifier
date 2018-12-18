
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
import json
import argparse 
from PIL import Image,ImageOps 

parser = argparse.ArgumentParser()

parser.add_argument("-i","--imgpath",help = "Path of the Image", metavar ='', type = str)
parser.add_argument("-d","--device",help= "Device to run cuda/cpu", metavar='',type = str)
parser.add_argument("-c","--checkpoint",help= "Check point to run on", metavar='',type = str)
parser.add_argument("-l","--hidden_layer",help= "First Hidden Layer", metavar='',type = int)
parser.add_argument("-k","--topk",help= "topk value", metavar='',type = int)
parser.add_argument("-j","--jsonfile",help= "jsonfile  value", metavar='',type = str)

args = parser.parse_args()

if args.imgpath:
    imgpath = args.imgpath
    print("Image path : {} ".format(imgpath))
else :
    print("Taking default image path")
    print("\t Please use <python predict -i image/path> as an argument")
    imgpath = "flowers/test/82/image_01686.jpg"
     
if args.checkpoint:
    checkpoint = args.checkpoint
    print("checkpoint as an argument : {}".format(checkpoint))
else :
    checkpoint = "checkpoint.pth"
    print("Taking system default checkpoint : {}".format(checkpoint))
    print("\t Try <python predict -c checkpoint/path>")

if args.device:
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else :
        device = "cpu"
    print("Device as argument : {} ".format(device))
else :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Taking system default device :")
    print("\tTry <python predict -d cuda> for cuda/cpu device ")

if args.topk:
    topk = args.topk
    print("topk as an argument : {}".format(topk))
else :
    topk = 5
    print("Taking system default topk : {}".format(topk)) 

if args.jsonfile:
    jsonfile = args.jsonfile
    print("json file as an argument : {}".format(jsonfile))
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

def load_checkpoint(filepath):
    #Loading checkpoint
    checkpoint = torch.load(filepath)
    #Assigning Model 
    network = checkpoint['model_type']
    if network == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif network == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif network == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif network == 'densenet121':
        model = models.densenet121(pretrained=True)
    else :
        print("No trained model found")
   
    print("Model trained is : {}".format(network))
    input_size = checkpoint['input_size']  
    output_size = checkpoint['output_size']
    if args.hidden_layer:
        hidden_layer = args.hidden_layer
        print("hidden_layers as an argument : {}".format(hidden_layer))
    else :
        hidden_layer = checkpoint['hidden_layer']
        print("Taking system default hidden_layers : {}".format(hidden_layer)) 
    
    classifier1 = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_layer, 2048)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(p=0.2)),
                          ('fc3',nn.Linear(2048, output_size)), 
                          ('output', nn.LogSoftmax(dim=1))        
                          ]))
    
    model.classifier = classifier1
    model.load_state_dict(checkpoint['state_dict'])
    #class_to_idx = None
    if args.jsonfile:
        class_to_idx = cat_to_name   
    else :
        class_to_idx = checkpoint['class_to_idx']
    return model,class_to_idx

print("Loading trained Model ")
model,class_to_idx = load_checkpoint(checkpoint)
model.to(device)
#print(model1)

def process_image(image):
    test_image_transform = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    test_image = Image.open(image)
    img_tensor = test_image_transform(test_image)
    img  = np.array(img_tensor)
    #print(img.shape)
    return img
    
testimg = process_image(imgpath)

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0)) 
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

#imshow(testimg)

def predict(image_path, model, topk):
    image = process_image(image_path)
    image = torch.from_numpy(image)
    inputs = image.float()
    inp = torch.unsqueeze(inputs,0)
    inp = inp.to(device)
    log_ps = model(inp)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p,top_class

print("Predicting input image")
probs, classes = predict(imgpath, model,int(topk))

probs, classes = probs.cpu(), classes.cpu()
prob = probs.detach().numpy()
clas = classes.detach().numpy()
#print(prob[0][1])
#print(clas[0][0])
print("########Prediction########")
for i,j in zip(clas,prob):
    for m,n in zip(i,j):
        print("{} :: {}".format(class_to_idx[str(m)],n))
