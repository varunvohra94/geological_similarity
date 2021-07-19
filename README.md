# Geological Similarity

## 1. Set up Hosted Kubeflow Pipelines

I will use a Kubernetes cluster to run the ML Pipelines.

### 1a. Start Hosted Pipelines
Create Hosted Kubeflow Pipelines Instance

* From the GCP Console, navigate to AI Platform | Pipelines and select +New Instance or browse directly to https://console.cloud.google.com/marketplace/details/google-cloud-ai-platform/kubeflow-pipelines
* Make sure your GCP project is selected in the dropdown at the top.
* Click CONFIGURE
* Change the App Instance Name to “kfpdemo”
* Click on the Create Cluster button and wait 2-3 minutes for cluster to get created.
* Click Deploy
* Navigate to https://console.cloud.google.com/ai-platform/pipelines/clusters
* Click on the HOSTED PIPELINES DASHBOARD LINK for kfpdemo on the cluster that you just started.

### 1b. Give the cluster permissions to run AI-Platfrom,App Engine etc.

In CloudShell:
Run:
```
git clone https://github.com/varunvohra94/geological_similarity.git
cd geological_similaritiy/
./setup_auth.sh varunmle us-central1-a cluster-1 default
```
* The first parameter is the service account you are using. I'm calling it kfpdemo
* The next three parameters are the zone, cluster, and namespace of the Kubeflow cluster -- you can find the cluster details at https://console.cloud.google.com/ai-platform/pipelines/clusters
* Check the service account IAM permissions at https://console.cloud.google.com/iam-admin/iam ; you can add other necessary permissions there.

## 2 Pipeline Description

I’m using Kubeflow on a Kubernetes cluster to orchestrate my pipeline

The Pipeline consists of three components:

* Unzip-and-upload-images
* Train-model
* Deploy app

### Unzip-and-upload-images:

The unzip-and-upload-images takes as input the location of the zipped folder in GCP and asks for a target bucket where the extracted images will be stored. This step also divides our images into a Train Set and Test set. This component then passes over the bucker name where the images were extracted

### Train-Model:

The train-model component takes as input the output of the last component (unzip-and-upload-images) which is the bucket where the train and test sets have been stored. The train-model component then submits a training job to GCP AI-Platform notebooks with the corresponding train and test sets. After training, the model and the embeddings for all the images are stored in the same bucket where the train and test images reside. The output of this component is the bucket where the model and the embeddings files are stored. 

### Deploy-App:

The deploy-app component takes as input the output of the train-model component (train-model) which is the bucket where the model and embeddings are stored. The deploy-app then builds an app on GCP App Engine 

### prediction Results:

The prediction results are available at 
https://opportune-baton-267215.uc.r.appspot.com/
This URL requires you to upload an image and select a value for K. Once done, the app should show you K most similar images to model.

### CI/CD:

I’m using GCP Cloud Build to enable CI/CD in my pipeline. I have linked my GitHub repository to cloud build and also have a github trigger in my repository. I also have cloudbuild.yaml files for all the containers in my application which automatically builds and pushes the containers to GCP container registry to always use the code from the latest Push to the ‘main’ branch. 
