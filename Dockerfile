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

# Install Chronos from Amazon via pip using their github repo
pip install git+https://github.com/amazon-science/chronos-forecasting.git

# Change the working directory
WORKDIR /usr/src/app/tradmodels

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "EvaluateModel.py","--dataset=ETTh1", "--model=ARIMA"]










# Make the shell script executable
#RUN echo "changing permission"
#RUN chmod +x scripts/long_term_forecast/*/*.sh

#RUN echo "running the code"
# Command to execute the shell script
#CMD ./scripts/long_term_forecast/$DIR_NAME/$SCRIPT_NAME



#RUN chmod +x scripts/long_term_forecast/M5/Autoformer_M5.sh

#CMD ["./scripts/long_term_forecast/M5/Autoformer_M5.sh"]
