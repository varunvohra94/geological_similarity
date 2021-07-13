FROM python:3.7-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/*

RUN pip install numpy gcsfs pandas matplotlib tensorboard google-cloud-storage

RUN mkdir /aws_mle && \
    cd /aws_mle && \
    git clone https://github.com/varunvohra94/geological_similarity.git

COPY train.sh ./

ENTRYPOINT ["bash","./train.sh"]