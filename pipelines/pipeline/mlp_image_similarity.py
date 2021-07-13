import kfp.dsl as dsl
from kfp.compiler import Compiler

@dsl.pipeline(
    name='image-similarity',
    description='Unzip and Upload Images'
)
def preprocess(project,bucket_in,file_path,bucket_out,mode):
    #Step 1: Unzip the images and upload to GCS
    unzip_and_upload = dsl.ContainerOp(
        name='Unzip and Upload Images',
        image='us.gcr.io/opportune-baton-267215/image-similarity-pipeline-zip-to-gcs',
        arguments=[
            '--project', project,
            '--bucket_in', bucket_in,
            '--file_path', file_path,
            '--bucket_out', bucket_out,
            '--mode',mode
        ],
        file_outputs={'output':'/output.txt'}
    )
Compiler().compile(preprocess,'mlp_image_similarity.tar.gz')