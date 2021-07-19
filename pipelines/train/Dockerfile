FROM google/cloud-sdk:latest

RUN mkdir /similarities && \
    cd similarities && \
    git clone https://github.com/varunvohra94/geological_similarity.git

COPY train_on_gcp.sh ./

ENTRYPOINT ["bash","./train_on_gcp.sh"]