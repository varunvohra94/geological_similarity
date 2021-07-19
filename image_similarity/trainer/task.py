import argparse
import json
import os

from . import model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help='GCS Path to Training file metadata',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        default='not_needed'
    )
    parser.add_argument(
        '--num_epochs',
        help= 'Number of epochs to run the training step for',
        type = int,
        default = 30
    )
    parser.add_argument(
        '--batch_size',
        help='Number of images for one iteration of train step',
        type=int,
        default=1024,
    )
    parser.add_argument(
        '--nn_size',
        help='Hidden Layers Sizes to use for CNN -- provide space-separated layers',
        nargs = '+',
        type = int,
        default=[32,64]
    )
    parser.add_argument(
        '--embed_size',
        help='Size of embeddings to compute similarity',
        type=int,
        default=128
    )
    parser.add_argument(
        '--learning_rate',
        help='Learning Rate for NN Model',
        type=float,
        default=0.001
    )
    args = parser.parse_args()
    arguments = args.__dict__

    bucket = arguments.pop('bucket')
    model.NUM_EPOCHS = arguments.pop('num_epochs')
    model.BATCH_SIZE = arguments.pop('batch_size')
    model.COV_NN_SIZE = arguments.pop('nn_size')
    model.EMBED_SIZE = arguments.pop('embed_size')
    model.LEARNING_RATE = arguments.pop('learning_rate')

    model.train_model(bucket)