import streamlit as st
import io
import matplotlib.image as mpimg
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
import numpy as np
import pandas as pd
from google.cloud import storage
import os
import gcsfs
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Geological Image Similarity"
)

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

def download_model(bucket,path):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(os.path.join(path,'model','model.pt'))
    dest = 'model.pt'
    blob.download_to_filename(dest)
    return dest

@st.cache
def load_model_and_embeddings(model_local,bucket,path,fs):
    model = torch.load(model_local)
    img_list_path = os.path.join('gs://',bucket,path,'model','embeddings','img_list.csv')
    embeddings_path = os.path.join('gs://',bucket,path,'model','embeddings','image_embeddings.npy')
    img_list = list(pd.read_csv(img_list_path)['img_loc'])
    with fs.open(embeddings_path,'rb') as f:
        embeddings = np.load(f)
    return model, embeddings, img_list


def read_image_upload(byte_img):
    byte_img = io.BytesIO(byte_img)
    upload_img = mpimg.imread(byte_img,0)
    return upload_img

def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5080, 0.5196, 0.5195],
                             std=[0.1852, 0.1995, 0.2193])
    ])
    return transform

def get_embedding(img, model, transform):
    img = transform(img)
    img = img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        emb, _ = model(img)
    return emb.numpy()


model_dir = os.environ.get('BUCKET')
bucket = model_dir.split('/',1)[0]
path = model_dir.split('/',1)[1]
model_local = download_model(bucket,path)
fs = gcsfs.GCSFileSystem()
model, embeddings, img_list = load_model_and_embeddings(model_local,bucket,path,fs)
transform = get_transform()


file_upload = st.sidebar.file_uploader("Choose a file",type=['jpg','png'])
if file_upload:
    file = file_upload.read()
    upload_img = read_image_upload(file)
    st.subheader('Anchor Image')
    st.image(upload_img,width=300)
st.markdown("---")

if file_upload:
    test_embedding = get_embedding(upload_img,model,transform)
    similarities = cosine_similarity(test_embedding,embeddings)


k = int(st.sidebar.number_input("Enter a value for K (Number of Similar Images to Show)",min_value=1,max_value=100,step=1,value=7))
st.markdown(f"Showing {k} simlar image(s)")

if file_upload:
    ids = np.flip(similarities.argsort()[0])[1:k+1]
    for index in ids:
        img = img_list[index]
        with fs.open(img,'rb') as f:
            img = mpimg.imread(f,0)
        st.text(f"{os.path.join(img_list[index].split('/')[-2], img_list[index].split('/')[-1])}")
        st.image(img,width=100)