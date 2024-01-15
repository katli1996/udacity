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

parser = argparse.ArgumentParser (description = 'Parser _ prediction script')


parser.add_argument ('image_dir', help = 'Input image path. Mandatory', type = str)
parser.add_argument ('--load_dir', help = 'Checkpoint path. Optional', default = "checkpoint.pth", type = str)
parser.add_argument ('--top_k', help = 'Choose number of Top K classes. Default is 5', default = 5, type = int)
parser.add_argument ('--category_names', help = 'Provide path of JSON file mapping categories to names. Optional', type = str)
parser.add_argument ('--GPU', help = "Input GPU if you want to use it", type = str)

args = parser.parse_args ()
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
    
def load_model(path):
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model
 def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    image = img_transforms(img)
    
    return image

def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img = img.to(device)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        ps = torch.exp(model(img))
        
    idx_to_flower = {v:cat_to_name[k] for k, v in model.class_to_idx.items()}    
    
    top_ps, top_classes = ps.topk(topk, dim=1)
    predicted_flowers = [idx_to_flower[i] for i in top_classes.tolist()[0]]
    
    print("This flower is {} with a probability of: {:.2f}%.".format(predicted_flowers[l], top_ps.tolist()[0]*100))

    return top_ps.tolist()[0], predicted_flowers

if __name__== "__main__":
    main()
