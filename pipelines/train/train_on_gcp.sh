#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
  echo "Usage ./train.sh bucket-name"
  exit
fi

BUCKET=$1
RUNTIME_VERSION=1.15
REGION=us-central1
JOBNAME=geological_similarity_$(date -u +%y%m%d_%H%M%S)

CODEDIR=/similarities/geological_similarity
OUTDIR=${BUCKET}

JOBNAME=geological_similarity_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME

gcloud ai-platform jobs submit training $JOBNAME \
  --region=$REGION \
  --module-name=trainer.task \
  --package-path=${CODEDIR}/image_similarity/trainer \
  --job-dir=gs://$OUTDIR \
  --staging-bucket=gs://$BUCKET \
  --scale-tier=PREMIUM_1 \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=3.7 \
  --stream_logs
  -- \
  --bucket=${BUCKET}

echo $OUTDIR > /output.txt