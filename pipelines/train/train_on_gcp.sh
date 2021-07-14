#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
  echo "Usage ./train.sh bucket-name"
  exit
fi

BUCKET=$1
RUNTIME_VERSION=1.15
REGION=us-central1

TRAIN_METADATA=gs://${BUCKET}/metadata/train.csv
TEST_METADATA=gs://${BUCKET}/metadata/test.csv

CODEDIR=/image_similarity/geological_similarity
OUTDIR=${BUCKET}



echo $OUTDIR $REGION




#pip install -r ${CODEDIR}/pipelines/train/requirements.txt
#
#export PYTHONPATH=${CODEDIR}/image_similarity:${PYTHONPATH}
#
#python -m trainer.task \
#  --job-dir=${OUTDIR} \
#  --train_metadata=${TRAIN_METADATA} \
#  --output_dir=${OUTDIR} \
#  --test_metadata=${TEST_METADATA}

echo $OUTDIR > /output.txt