import argparse
import json
import os

from . import model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_metadata',
        help='GCS Path to Training file metadata',
        required=True
    )
    parser.add_argument(
        '--test_metadata',
        help='GCS Path to Testing file metadata',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS Location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--job_dir',
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

    output_dir = arguments.pop('output_dir')
    model.TRAIN_METADATA = arguments.pop('train_metadata')
    model.TEST_METADATA = arguments.pop('test_metadata')
    model.NUM_EPOCHS = arguments.pop('num_epochs')
    model.BATCH_SIZE = arguments.pop('batch_size')
    model.COV_NN_SIZE = arguments.pop('nn_size')
    model.EMBED_SIZE = arguments.pop('embed_size')
    model.LEARNING_RATE = arguments.pop('learning_rate')

    model.train_model(output_dir)