FROM FROM python:3.7-slim
RUN mkdir /aws_mle && \
    cd /aws_mle && \
    git clone https://github.com/varunvohra94/geological_similarity.git
COPY requirement.txt ./
RUN pip install -r requirement.txt
COPY train.sh ./

ENTRYPOINT ["bash","./train.sh"]