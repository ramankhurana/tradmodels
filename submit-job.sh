#!/bin/bash

## config map needs to be created before submitting the job.
## one config map is needed for a scan, or setup environment variables
kubectl create configmap datasetschema-config --from-file=datasetschema.yaml -n refit-release

templateyaml=cluster-run/job-tiger.yaml
# Define datasets and models
#declare -a datasets=("ETTh1" "ETTh2" "ETTm1 ETTm2")
declare -a datasets=("ETTh1")
declare -a models=("ARIMA")

# Loop through datasets and models
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do

    # Create a sanitized version of dataset and model names for use in Kubernetes object names, it should be in lower case
    clean_dataset=$(echo $dataset | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')
    clean_model=$(echo $model | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')

    job_name="tradmodels-${clean_model}-${clean_dataset}-$(date +%Y%m%d%H%M%S)"
    echo "submitting job for:" $job_name

    # Replace the dataset and model names in a job template and apply it
    sed -e "s/TEMPLATE_DATASET_NAME/$dataset/g" \
        -e "s/TEMPLATE_MODEL_NAME/$model/g" \
        -e "s/tradmodels-job/$job_name/g" $templateyaml | kubectl apply -f -

  done
done
