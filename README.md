# Geological Similarity

## 1. Set up Hosted Kubeflow Pipelines

You will use a Kubernetes cluster to run the ML Pipelines.

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

