sed -i '' 's/horizon: 336/horizon: 720/' datasetschema.yaml
python EvaluateModel.py --dataset=ETTh1 --model=TimeGPT
python EvaluateModel.py --dataset=ETTh2 --model=TimeGPT
python EvaluateModel.py --dataset=ETTm1 --model=TimeGPT
python EvaluateModel.py --dataset=ETTm2 --model=TimeGPT
python EvaluateModel.py --dataset=Electricity --model=TimeGPT
python EvaluateModel.py --dataset=Exchange --model=TimeGPT
python EvaluateModel.py --dataset=Weather --model=TimeGPT
python EvaluateModel.py --dataset=Illness --model=TimeGPT
python EvaluateModel.py --dataset=Divvy --model=TimeGPT
python EvaluateModel.py --dataset=M5 --model=TimeGPT
