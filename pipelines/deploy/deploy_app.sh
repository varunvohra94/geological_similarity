#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./deploy.sh Bucket"
    exit
fi

export BUCKET=$1

cd /image_similarity/geological_similarity/serving

gcloud app deploy

PROJECT=$(gcloud config get-value project)
echo "https://${PROJECT}.appspot.com" > appurl.txt
