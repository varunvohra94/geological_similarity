import pandas as pd
import gcsfs
import numpy as np
import matplotlib.image as mpimg
import os

from sklearn.metrics import accuracy_score
from google.cloud import storage

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from . import utils

NUM_EPOCHS = 30
BATCH_SIZE = 1024
COV_NN_SIZE = [32,64]
EMBED_SIZE = 128
LEARNING_RATE = 0.001


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


class GeologicalDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, step, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.step = step
        self.transform = transform

        img_names_list = [os.path.join(path, name)
                          for path, _, files in os.walk(os.path.join(os.getcwd(), self.step))
                          for name in files]

        labels = os.listdir(os.path.join(os.getcwd(), self.step))
        labels = sorted(labels)
        labels_to_idx = {k: v for v, k in enumerate(labels)}

        labels_list = [labels_to_idx[os.path.join(path, name).split('/')[-2]]
                       for path, _, files in os.walk(os.path.join(os.getcwd(), self.step))
                       for name in files
                       ]

        self.img_names = img_names_list
        self.labels = labels_list

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = mpimg.imread(self.img_names[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label


# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     """Uploads a file to the bucket."""
#     # The ID of your GCS bucket
#     # bucket_name = "your-bucket-name"
#     # The path to your file to upload
#     # source_file_name = "local/path/to/file"
#     # The ID of your GCS object
#     # destination_blob_name = "storage-object-name"
#
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print(
#         "File {} uploaded to {}.".format(
#             source_file_name, destination_blob_name
#         )
#     )

# def get_img_and_label_list(metadata):
#     df = pd.read_csv(metadata)
#     df['_label'] = pd.factorize(df.label)[0]
#     return list(df['img_loc']),list(df['_label'])

# def get_transform(step):
#     if step == 'Train':
#         transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(0.5),
#             transforms.RandomVerticalFlip(0.5),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5080, 0.5196, 0.5195],
#                                  std=[0.1852, 0.1995, 0.2193])
#         ])
#     else:
#         transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5080, 0.5196, 0.5195],
#                              std=[0.1852, 0.1995, 0.2193])
#         ])
#     return transform

def run_training(train_loader,test_loader,output_dir):
    model = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):

        train_accuracy_epoch = []
        for train_batch_idx, (images, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            _, outputs = model(images)
            predictions = torch.max(outputs, 1)[1]
            train_accuracy = accuracy_score(predictions, labels)
            train_accuracy_epoch.append(train_accuracy)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

        avg_train_accuracy_epoch = np.mean(train_accuracy_epoch)
        writer.add_scalar("Train_accuracy", avg_train_accuracy_epoch, epoch)

        # At the end of epoch - evaluate test set accuracy
        test_accuracy_epoch = []
        for test_batch_idx, (images, labels) in enumerate(test_loader):
            model.eval()
            _, outputs = model(images)
            predictions = torch.max(outputs, 1)[1]
            test_accuracy = accuracy_score(predictions, labels)
            test_accuracy_epoch.append(test_accuracy)

        avg_test_accuracy_epoch = np.mean(test_accuracy_epoch)
        writer.add_scalar("Test_accuracy", avg_test_accuracy_epoch, epoch)

        print('Epoch: ', epoch,
              '  Train accuracy:', round(avg_train_accuracy_epoch, 4),
              '  Test accuracy:', round(avg_test_accuracy_epoch, 4)
              )
    writer.flush()
    local_model_path = 'model.pt'
    torch.save(model, local_model_path)
    # upload_model(local_model_path, output_dir)
    # bucket = output_dir.split('/')[0]
    # blob_name = output_dir.split('/')[1] + '/model/model.pt'
    # upload_blob(bucket,'model.pt',blob_name)
    return model, local_model_path

# def upload_model(path, output_dir):
#     bucket = output_dir.split('/')[0]
#     # blob_name = output_dir.split('/')[1] + '/model/model.pt'
#     blob_name = os.path.join(os.path.join(output_dir.split('/')[1], 'model'),path)
#     upload_blob(bucket,path,blob_name)

# def get_all_images(df_train,df_test):
#     df_train = pd.read_csv(df_train)
#     df_test = pd.read_csv(df_test)
#     df = pd.concat([df_train,df_test])
#     img_files = list(df['img_loc'])
#     df['img_loc'].to_csv(os.path.join())
#     return img_files
#
# def get_embedding_gcs(file, fs, model,test_transform):
#     with fs.open(file,'rb') as f:
#         image = mpimg.imread(f,0)
#     image = test_transform(image)
#     image = image.unsqueeze(0)
#     model.eval()
#     with torch.no_grad():
#         emb, _ = model(image)
#     return emb.numpy()

def get_embedding(file, model, transform):
    image = mpimg.imread(file)
    image = transform(image)
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        emb, _ = model(image)
    return emb.numpy()

# def save_embeddings(img_files,image_embeddings,output_dir):
#     fs = gcsfs.GCSFileSystem()
#     gcs_loc = os.path.join('gs://',output_dir,'model','embeddings')
#     img_files = [os.path.join('gs://',
#                               output_dir,
#                               i.split('/')[-3],
#                               i.split('/')[-2],i.split('/')[-1])
#                  for i in img_files]
#     pd.DataFrame(img_files, columns=['img_loc']).to_csv(os.path.join(gcs_loc,'img_list.csv'),index=False)
#     with fs.open(os.path.join(gcs_loc,'embeddings.npy'),'wb') as f:
#         np.save(f,image_embeddings)
#     return gcs_loc
#
# def download_images(path,step):
#     bucket = path.split('/',1)[0]
#     prefix = os.path.join(path.split('/',1)[1],step)
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket)
#     blobs = bucket.list_blobs(prefix=prefix)
#     for blob in blobs:
#         class_name = blob.name.rsplit('/',2)[1]
#         file_name = blob.name.rsplit('/',1)[1]
#         try:
#             os.makedirs(os.path.join(step,class_name))
#         except:
#             pass
#         blob.download_to_filename(os.path.join(os.path.join(step,class_name),file_name))

def train_model(output_dir):
    datasets = {}
    data_loaders = {}
    img_loc_step = {}
    transform_step = {}
    for step in ['Train','Test']:
        utils.download_images(output_dir,step)
        transform_step[step] = utils.get_transform(step)
        img_loc_step[step] = [os.path.join(path, name)
                              for path, _, files in os.walk(os.path.join(os.getcwd(),step))
                              for name in files]
        datasets[step] = GeologicalDataset(step,transform_step[step])
        data_loaders[step] = torch.utils.data.DataLoader(datasets[step],batch_size=BATCH_SIZE)


        # train_set = GeologicalDataset(train_img_list,train_label_list,train_transform)
        # test_set = GeologicalDataset(test_img_list,test_label_list,test_transform)
        #
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
        # test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

    model, local_model_path = run_training(data_loaders['Train'], data_loaders['Test'], output_dir)
    utils.upload_model(local_model_path, output_dir)
    # img_files = get_all_images(TRAIN_METADATA,TEST_METADATA)
    img_files = sum(img_loc_step.values(), [])
    # fs = gcsfs.GCSFileSystem()
    image_embeddings = np.zeros((len(img_files), 128))
    for idx, file in enumerate(img_files):
        image_embeddings[idx] = get_embedding(file, model, transform_step['Test'])

    utils.save_embeddings(img_files, image_embeddings, output_dir)

