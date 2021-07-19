REPO_NAME=geological_similarity
REPO_OWNER=VarunVohra

#for trigger_name in trigger-000 trigger-001 trigger-002 trigger-003; do
#  gcloud beta builds triggers delete --quiet $trigger_name
#done


create_github_trigger() {
    DIR_IN_REPO=$(pwd | sed "s%${REPO_NAME}/% %g" | awk '{print $2}')
    gcloud beta builds triggers create github \
      --build-config="${DIR_IN_REPO}/cloudbuild.yaml" \
      --included-files="${DIR_IN_REPO}/**" \
      --branch-pattern="^main$" \
      --repo-name=${REPO_NAME} --repo-owner=${REPO_OWNER}
}

for container_dir in $(ls -d */ | sed 's%/%%g'); do
    cd $container_dir
    create_github_trigger
    cd ..
done