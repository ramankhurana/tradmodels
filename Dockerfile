## This docker file needs two inputs from environment; directory name and bash script name
## This is provided via k8 job yaml file

# python image used for Extformer
FROM python:3.10.1

# set working dir
WORKDIR /usr/src/app

# install git
RUN apt-get update && apt-get install -y git

# Clone repository from GitHub
RUN git clone https://github.com/ramankhurana/tradmodels.git


# Change the working directory
WORKDIR /usr/src/app/tradmodels

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Install Chronos from Amazon via pip using their github repo
RUN pip install git+https://github.com/amazon-science/chronos-forecasting.git

# Install transformers to use its CLI for downloading models
RUN pip install transformers

# Download the Chronos T5 model using transformers CLI
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('amazon/chronos-t5-tiny')"


CMD ["python", "EvaluateModel.py","--dataset=ETTh1", "--model=ARIMA"]










# Make the shell script executable
#RUN echo "changing permission"
#RUN chmod +x scripts/long_term_forecast/*/*.sh

#RUN echo "running the code"
# Command to execute the shell script
#CMD ./scripts/long_term_forecast/$DIR_NAME/$SCRIPT_NAME



#RUN chmod +x scripts/long_term_forecast/M5/Autoformer_M5.sh

#CMD ["./scripts/long_term_forecast/M5/Autoformer_M5.sh"]
