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
    local_model_path = 'model.pt'
    torch.save(model, local_model_path)
    return model, local_model_path


def get_embedding(file, model, transform):
    image = mpimg.imread(file)
    image = transform(image)
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        emb, _ = model(image)
    return emb.numpy()

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

    model, local_model_path = run_training(data_loaders['Train'], data_loaders['Test'])
    utils.upload_model(local_model_path, output_dir)
    # img_files = get_all_images(TRAIN_METADATA,TEST_METADATA)
    img_files = sum(img_loc_step.values(), [])
    # fs = gcsfs.GCSFileSystem()
    image_embeddings = np.zeros((len(img_files), 128))
    for idx, file in enumerate(img_files):
        image_embeddings[idx] = get_embedding(file, model, transform_step['Test'])

    utils.save_embeddings(img_files, image_embeddings, output_dir)

