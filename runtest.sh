sed -i '' 's/horizon: 720/horizon: 6/' datasetschema.yaml
python EvaluateModel.py --dataset=ETTh1 --model=ARNET
python EvaluateModel.py --dataset=ETTh2 --model=ARNET
python EvaluateModel.py --dataset=ETTm1 --model=ARNET
python EvaluateModel.py --dataset=ETTm2 --model=ARNET
python EvaluateModel.py --dataset=Electricity --model=ARNET
python EvaluateModel.py --dataset=Exchange --model=ARNET
python EvaluateModel.py --dataset=Weather --model=ARNET
python EvaluateModel.py --dataset=Illness --model=ARNET
python EvaluateModel.py --dataset=Divvy --model=ARNET
python EvaluateModel.py --dataset=M5 --model=ARNET
