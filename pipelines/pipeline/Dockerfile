FROM python:3.8

# Install gcloud sdk
RUN apt-get update -y && apt-get install --no-install-recommends -y -q ca-certificates wget unzip git

RUN pip install google-api-python-client

RUN cd / && \
    wget -nv https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.zip && \
    unzip -qq google-cloud-sdk.zip -d tools && \
    rm google-cloud-sdk.zip && \
    tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    tools/google-cloud-sdk/bin/gcloud -q components update \
        gcloud core gsutil && \
    tools/google-cloud-sdk/bin/gcloud config set component_manager/disable_update_check true && \
    touch /tools/google-cloud-sdk/lib/third_party/google.py

ENV PATH $PATH:/tools/node/bin:/tools/google-cloud-sdk/bin


# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.
COPY requirements.txt ./

# Install production dependencies.
RUN pip install -r requirements.txt

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Run the web service on container startup.
# Use gunicorn webserver with one worker process and 2 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 cloudrunapp:app