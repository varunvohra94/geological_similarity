FROM google/cloud-sdk:latest

RUN mkdir /similarities && \
    cd similarities && \
    git clone https://github.com/varunvohra94/geological_similarity.git

COPY deploy_app.sh ./

ENTRYPOINT ["bash","./deploy.sh"]