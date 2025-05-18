#sed -i '' 's/horizon: 6/horizon: 12/' datasetschema.yaml
python EvaluateModel.py --dataset=ETTh1 --model=ARIMA
python EvaluateModel.py --dataset=ETTh2 --model=ARIMA
python EvaluateModel.py --dataset=ETTm1 --model=ARIMA
python EvaluateModel.py --dataset=ETTm2 --model=ARIMA
python EvaluateModel.py --dataset=Exchange --model=ARIMA
python EvaluateModel.py --dataset=Weather --model=ARIMA
python EvaluateModel.py --dataset=Illness --model=ARIMA
python EvaluateModel.py --dataset=Divvy --model=ARIMA
python EvaluateModel.py --dataset=M5 --model=ARIMA
python EvaluateModel.py --dataset=Electricity --model=ARIMA
