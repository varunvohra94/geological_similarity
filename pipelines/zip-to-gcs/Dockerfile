FROM python:3.7-slim
RUN mkdir /image-similarity
RUN pip install -U google-cloud-storage pandas gcsfs
COPY unzip_to_gcs.py /image-similarity
ENTRYPOINT ["python","/image-similarity/unzip_to_gcs.py"]
