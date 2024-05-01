There are various parts in this part of the repository which will contorl the evaluation of various models.

For most models we are relying on the darts and neuralprophet, and the datasets are prepared before feeding them to the respective models.

For some cases, datasets needs to be different format and hence it is pre-processed in advance to make it compatible with the model.

The schemas needed for this evaluation are:

- Run Schema
- Dataset Schema
- Model Schema


runschema.yaml: Run schema can control which model to run, and then which datasets are required to be run for this model. By default a given set of daatsets are processed, however depending on the need, some of them can be masked using the mask flag. runmodel: decides if the model needs to run or not. 


datasetschema.yaml: this schema has the dataset name and path to data itself. It also has the information about lag, horizon and time range for train,val,test window. date_col should be mentioned so that it can be ignored for forcasting. 


dataReader.py: This file reads the dataset. 