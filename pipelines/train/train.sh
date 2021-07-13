#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
  echo "Usage ./train.sh bucket-name"
  exit
fi

BUCKET=$1
VERSION=1.15
REGION=us-central1
TRAIN_METADATA=gs://${BUCKET}/metadata/train.csv
TEST_METADATA=gs://${BUCKET}/metadata/test.csv
CODEDIR=/aws_mle/geological_similarity/
OUTDIR=gs://${BUCKET}/model

export PYTHONPATH=${CODEDIR}/image_similarity:${PYTHONPATH}

python -m trainer.task \
  --job-dir=${OUTDIR} \
  --train_metadata=${TRAIN_METADATA} \
  --output_dir=${OUTDIR} \
  --test_metadata=${TEST_METADATA}

echo $OUTDIR > /output.txt