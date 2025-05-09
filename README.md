There are various parts in this part of the repository which will contorl the evaluation of various models.

For most models we are relying on the darts and neuralprophet, and the datasets are prepared before processing for a given model. Most models requires different data style/type in order to process dataset.
Therefore an abstraction layer is added before data is processed. 

For some cases, datasets needs to be different format and hence it is pre-processed in advance to make it compatible with the model.

The schemas needed for this evaluation are:

- Run Schema
- Dataset Schema
- Model Schema


runschema.yaml: Run schema can control which model to run, and then which datasets are required to be run for this model. By default a given set of daatsets are processed, however depending on the need, some of them can be masked using the mask flag. runmodel: decides if the model needs to run or not. 


datasetschema.yaml: this schema has the dataset name and path to data itself. It also has the information about lag, horizon and time range for train,val,test window. date_col should be mentioned so that it can be ignored for forcasting. 

## EvaluateModel.py
This is the script that evaluate the model.
It requires two yaml file to initialize the evaluate class and then evaluate the model, save results and print them using following set of python code
```
    evaluator = EvaluateModel('datasetschema.yaml', 'runschema.yaml', dataset, model)
    evaluator.evaluate()
    #evaluator.save_predictions()                                                                                                                                                                                        evaluator.save_results_to_csv()
    evaluator.printresults()
```


dataReader.py: This file reads the dataset. not sure if it is still needed. 