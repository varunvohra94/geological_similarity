#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
  echo "Usage ./train.sh bucket-name"
  exit
fi

BUCKET=$1
RUNTIME_VERSION=1.15
REGION=us-central1

CODEDIR=/similarities/geological_similarity
OUTDIR=${BUCKET}

JOBNAME=geological_similarity_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME

gcloud ai-platform jobs submit training $JOBNAME \
  --region=$REGION \
  --module-name=trainer.task \
  --package-path=${CODEDIR}/image_similarity/trainer \
  --job-dir=$OUTDIR \
  --staging-bucket=gs://$BUCKET \
  --scale-tier=PREMIUM_1 \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=3.7 \
  -- \
  --bucket=${BUCKET}

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