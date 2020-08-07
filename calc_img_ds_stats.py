import argparse
import torch
from torchvision import datasets, transforms as T
from pathlib import Path
 
parser = argparse.ArgumentParser()
 
parser.add_argument("data_dir", help="Directory location of the images.")
 
args = parser.parse_args()
 
transform = T.ToTensor()
dataset = datasets.ImageFolder(str(Path(args.data_dir)), transform=transform)
 
mean = 0
std = 0
for img, _ in dataset:
    img = img.view(3, -1)
 
    mean += img.mean(1)
    std += img.std(1)
 
mean /= len(dataset)
std /= len(dataset)
 
print("mean: " + str(mean))
print("std dev: " + str(std))