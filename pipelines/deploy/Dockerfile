FROM google/cloud-sdk:latest

RUN mkdir /image_similarity && \
    cd image_similarity && \
    git clone https://github.com/varunvohra94/geological_similarity.git

COPY deploy.sh ./

ENTRYPOINT ["bash","./deploy.sh"]