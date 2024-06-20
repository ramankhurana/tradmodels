sed -i '' 's/horizon: 720/horizon: 6/' datasetschema.yaml
python EvaluateModel.py --dataset=ETTh1 --model=Chronos
python EvaluateModel.py --dataset=ETTh2 --model=Chronos
python EvaluateModel.py --dataset=ETTm1 --model=Chronos
python EvaluateModel.py --dataset=ETTm2 --model=Chronos
python EvaluateModel.py --dataset=Electricity --model=Chronos
python EvaluateModel.py --dataset=Exchange --model=Chronos
python EvaluateModel.py --dataset=Weather --model=Chronos
python EvaluateModel.py --dataset=Illness --model=Chronos
python EvaluateModel.py --dataset=Divvy --model=Chronos
python EvaluateModel.py --dataset=M5 --model=Chronos
