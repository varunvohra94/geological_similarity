from google.cloud import storage
import os
import gcsfs
import pandas as pd
import numpy as np
from torchvision import transforms

def download_images(path,split):
    bucket = path.split('/',1)[0]
    prefix = os.path.join(path.split('/',1)[1],split)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        class_name = blob.name.rsplit('/',2)[1]
        file_name = blob.name.rsplit('/',1)[1]
        try:
            os.makedirs(os.path.join(split,class_name))
        except:
            pass
        blob.download_to_filename(os.path.join(split,class_name) + '/' + file_name)

def get_transform(step):
    if step == 'Train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5080, 0.5196, 0.5195],
                                 std=[0.1852, 0.1995, 0.2193])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5080, 0.5196, 0.5195],
                             std=[0.1852, 0.1995, 0.2193])
        ])
    return transform

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def upload_model(path, output_dir):
    bucket = output_dir.split('/')[0]
    blob_name = os.path.join(output_dir.split('/')[1], 'model', path)
    upload_blob(bucket, path, blob_name)


def save_embeddings(img_files,image_embeddings,output_dir):
    fs = gcsfs.GCSFileSystem()
    gcs_loc = os.path.join('gs://', output_dir , 'model', 'embeddings')
    img_files = [os.path.join('gs://',
                              output_dir,
                              i.split('/')[-3],
                              i.split('/')[-2],
                              i.split('/')[-1])
                 for i in img_files]
    pd.DataFrame(img_files, columns=['img_loc']).to_csv(os.path.join(gcs_loc,'img_list.csv'),index=False)
    with fs.open(os.path.join(gcs_loc,'embeddings.npy'),'wb') as f:
        np.save(f,image_embeddings)
    return gcs_loc