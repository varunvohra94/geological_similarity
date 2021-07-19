import argparse
import google.cloud

def randomize_files(file_list):
    import random
    random.seed(42)
    random.shuffle(file_list)
    return file_list

def get_training_and_testing_sets(file_list,split):
    """
    Function to split the images into train and test set
    :param file_list: list of
    """
    from math import floor
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

def create_bucket_class_location(bucket_name,project):
    """
    Create a new bucket in specific location with storage class
    :param bucket_name: Name of bucket to be created
    :param project: GCP Project to create bucket in
    """
    from google.cloud import storage

    storage_client = storage.Client(project=project)

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = "COLDLINE"
    storage_client.create_bucket(bucket, location="us")
    return

def get_date_today():
    """
    Function to get date at runtime
    """
    from datetime import date
    return date.today().strftime('%d-%m-%Y')

def write_csv_to_gcs(data,bucket_out,today,is_train=True):
    """
    Function to write image location mtadata to GCP
    :param data: metadata to write to gcs
    :param bucket_out: bucket name to write metadata
    :param today: runtime date
    :param is_train: Boolean flag to upload either Train or Test metadata
    """
    import pandas as pd
    import gcsfs
    import os
    df = pd.DataFrame(data)

    if is_train:
        gcs_loc = os.path.join('gs://', bucket_out, today, 'metadata', 'train.csv')
        df.to_csv(gcs_loc, index=False)
    else:
        gcs_loc = os.path.join('gs://', bucket_out, today, 'metadata', 'test.csv')
        df.to_csv(gcs_loc, index=False)

def zipextract(in_test_mode,project,bucket_in,file_path,bucket_out,image_type,split):
    """
    :param in_test_mode: Boolean values (If True only extracts 100 train images and 30 test images)
    :param project: GCP Project to run on
    :param bucket_in: Bucket name containing .zip folder
    :param: file_path: Prefix for .zip folder
    :param: bucket_out: Bucket to save the extracted images at
    :image_type: Type of images
    :split: Train or Test Split
    :return: % Split between train and testing data
    """
    import os
    import io
    import json
    import logging
    from google.cloud import storage
    from zipfile import ZipFile
    from zipfile import is_zipfile
    today = get_date_today()

    storage_client = storage.Client(project=project)
    _bucket_in = storage_client.get_bucket(bucket_in)

    try:
        _bucket_out = storage_client.get_bucket(bucket_out)
    except:
        logging.info('Not Found, Creating Bucket')
        create_bucket_class_location(bucket_out,project)
        _bucket_out = storage_client.get_bucket(bucket_out)

    train_set = []
    test_set = []

    blob = _bucket_in.blob(file_path)
    zipbytes = io.BytesIO(blob.download_as_string())
    if is_zipfile(zipbytes):
        with ZipFile(zipbytes, 'r') as myzip:
            filenames = [i for i in myzip.namelist() if image_type in i]
            filenames = randomize_files(filenames)
            train,test = get_training_and_testing_sets(filenames,split)
            if in_test_mode:
                train = train[:100]
                test=test[:30]
            for content_filename in train:
                content_file = myzip.read(content_filename)
                img_name = content_filename.rsplit('/',1)[1]
                class_name = content_filename.rsplit('/',2)[1]
                blob = _bucket_out.blob(os.path.join(today,'Train',class_name,img_name))
                blob.upload_from_string(content_file)
                gcs_loc = os.path.join('gs://',bucket_out,today,'Train',class_name,img_name)
                train_set.append({'img_loc':gcs_loc,'label':class_name})
            for content_filename in test:
                content_file = myzip.read(content_filename)
                img_name = content_filename.rsplit('/',1)[1]
                class_name = content_filename.rsplit('/',2)[1]
                blob = _bucket_out.blob(os.path.join(today,'Test',class_name,img_name))
                blob.upload_from_string(content_file)
                gcs_loc = os.path.join('gs://', bucket_out, today, 'Test', class_name, img_name)
                test_set.append({'img_loc': gcs_loc, 'label': class_name})

    write_csv_to_gcs(train_set,bucket_out,today,True)
    write_csv_to_gcs(test_set,bucket_out,today,False)

    with open('/output.txt','w') as f:
        f.write(os.path.join(bucket_out,today))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',
                        type=str,
                        required=True,
                        help='The GCP Project')
    parser.add_argument('--bucket_in',
                        type=str,
                        required=True,
                        help='Bucket with stored data for training and testing')
    parser.add_argument('--file_path',
                        type=str,
                        required=True,
                        help='Full path of zipped images')
    parser.add_argument('--bucket_out',
                        type=str,
                        required=True,
                        help='Bucket with stored data for training and testing')
    parser.add_argument('--image_type',
                        type=str,
                        default='.jpg',
                        choices=['.jpg','.png']
                        )
    parser.add_argument('--train_split',
                        type=float,
                        default=0.9)
    parser.add_argument('--mode',
                        type=str,
                        default='test',
                        choices=['test','full'])
    args = parser.parse_args()

    zipextract(args.mode=='test',args.project,args.bucket_in,args.file_path,args.bucket_out,args.image_type,args.train_split)
