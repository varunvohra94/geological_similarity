import argparse
import torch
import torch.nn as nn
from google.cloud import storage
import os
import gcsfs
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.image as mpimg
import numpy as np
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

def download_model(model_path):
    bucket = model_path.split('gs://')[1].split('/',1)[0]
    path = os.path.join(model_path.split('gs://')[1].split('/',1)[1],'model_state_dict.pth')


    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket)

    blob = bucket.blob(path)
    dest = 'model_state_dict.pth'
    blob.download_to_filename(dest)
    return dest

def get_embedding(img,model,transform):
    img = transform(img)
    img = img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        emb, _ = model(img)
    return emb.numpy()


def get_similarities(model,img_list,embeddings,img,transform):
    K = 7
    test_embed = get_embeddings(img,model,transform)
    similarities = cosine_similarity(test_embed, image_embeddings)
    ids = np.flip(similarities.argsort()[0])[1:K+1]
    return ids

@st.cache
def load_model_and_embeddings(model_path,model_local,fs):
    model = CNN()
    model.load_state_dict(torch.load(model_local))
    img_list_path = os.path.join(model_path,'embeddings','img_list.csv')
    img_list = list(pd.read_csv(img_list_path)['img_loc'])
    embed_path = os.path.join(model_path,'embeddings','image_embeddings.npy')
    with fs.open(embed_path,'rb') as f:
        image_embeddings = np.load(f)
    return model,image_embeddings,img_list

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to model and embeddings')
    args = parser.parse_args()
    fs = gcsfs.GCSFileSystem()
    model_local = download_model(args.model_path)
    model,embeddings,img_list = load_model_and_embeddings(args.model_path,model_local,fs)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5080, 0.5196, 0.5195],
                             std=[0.1852, 0.1995, 0.2193])
    ])
    ids = get_similarities(model,img_list,embeddings,img,transform)