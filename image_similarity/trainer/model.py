import pandas as pd
import gcsfs
import numpy as np
import matplotlib.image as mpimg

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

TRAIN_METADATA = None
TEST_METADATA = None

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

    def __init__(self, img_names, labels, transform=None):
        self.transform = transform
        self.img_names = img_names
        self.labels = labels
        self.fs = gcsfs.GCSFileSystem()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        with self.fs.open(self.img_names[idx]) as f:
            image = mpimg.imread(f,0)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def get_img_and_label_list(metadata):
    df = pd.read_csv(metadata)
    df['_label'] = pd.factorize(df.label)[0]
    return list(df['img_loc']),list(df['_label'])

def get_train_transform():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        # transforms.RandomRotation(degrees=(0,360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5080, 0.5196, 0.5195],
                             std=[0.1852, 0.1995, 0.2193])
    ])
    return train_transform

def get_test_transform():
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5080, 0.5196, 0.5195],
                             std=[0.1852, 0.1995, 0.2193])
    ])
    return test_transform

def run_training(train_loader,test_loader):
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
    torch.save(model.state_dict(),'model_state_dict_path.pth')
    bucket = output_dir.split('/')[0]
    blob_name = output_dir.split('/')[1] + '/model/model_state_dict_path.pth'
    upload_blob(bucket,'model_state_dict_path.pth',blob_name)

def train_model(output_dir):
    train_img_list,train_label_list = get_img_and_label_list(TRAIN_METADATA)
    test_img_list,test_label_list = get_img_and_label_list(TEST_METADATA)

    train_transform = get_train_transform()
    test_transform = get_test_transform()

    train_set = GeologicalDataset(train_img_list,train_label_list,train_transform)
    test_set = GeologicalDataset(test_img_list,test_label_list,test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

    run_training(train_loader,test_loader)