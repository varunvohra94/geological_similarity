import kfp.dsl as dsl
from kfp.compiler import Compiler

@dsl.pipeline(
    name='image-similarity',
    description='Unzip and Upload Images'
)
def unzip_train_deploy(project,bucket_in,file_path,bucket_out,mode):
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
        file_outputs={'bucket':'/output.txt'}
    )

    train = dsl.ContainerOp(
        name='Train Model',
        image='us.gcr.io/opportune-baton-267215/image-similarity-pipeline-train',
        arguments=[
            unzip_and_upload.outputs['bucket']
        ],
        file_outputs={'bucket':'/output.txt'}
    )
    train.set_memory_request('2G')
    train.set_cpu_request('1')


Compiler().compile(unzip_train_deploy,'mlp_image_similarity.tar.gz')