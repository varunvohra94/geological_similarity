import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from model import MultiClassifier
import io
import streamlit as st

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.nn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(in_features=2304, out_features=512),
            nn.Dropout2d(0.25),
            nn.Linear(in_features=512, out_features=128),
        )

        self.linear = nn.Linear(in_features=128, out_features=6)

    def forward(self, x):
        embedding = self.nn(x)
        x = self.linear(embedding)
        return embedding, x

def download_model():


if __name__='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to model and embeddings')
