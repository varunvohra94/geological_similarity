import streamlit as st
import io
import matplotlib.image as mpimg
import tempfile
from deploy import *
import pandas as pd
import argparse
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title = 'Image Feature Prediction',
    page_icon = '😇'
)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',
                    type=str,
                    required=True,
                    help='Path to model and embeddings')
args = parser.parse_args()
fs = gcsfs.GCSFileSystem()
model_local = download_model(args.model_path)
model, embeddings, img_list = load_model_and_embeddings(args.model_path, model_local, fs)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5080, 0.5196, 0.5195],
                         std=[0.1852, 0.1995, 0.2193])
])


def saveImage(byteImage):
    bytesImg = io.BytesIO(byteImage)
    imgFile = mpimg.imread(bytesImg,0)

    return imgFile

st.header("Image Similarities")
file_upload = st.file_uploader("Choose a file",type=['jpg','png'])
K=7
if file_upload:
    file = file_upload.read()
    img = saveImage(file)
    st.image(img,width=300)
    test_embed = get_embedding(img,model,transform)
    similarities = cosine_similarity(test_embed, embeddings)
    ids = np.flip(similarities.argsort()[0])[1:K+1]
    for index in ids:
        image = img_list[index]
        with fs.open(image,'rb') as f:
            image=mpimg.imread(f,0)
        st.image(image,width=300)