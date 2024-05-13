#!/bin/bash
templateyaml=cluster-run/job-tiger.yaml
# Define datasets and models
declare -a datasets=("ETTh1")
declare -a models=("ARIMA")

# Loop through datasets and models
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
      # Replace the dataset and model names in a job template and apply it
      sed "s/TEMPLATE_DATASET_NAME/$dataset/g; s/TEMPLATE_MODEL_NAME/$model/g" $templateyaml | kubectl apply -f -
  done
done
