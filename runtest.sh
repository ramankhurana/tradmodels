#sed -i '' 's/horizon: 6/horizon: 12/' datasetschema.yaml
python EvaluateModel.py --dataset=ETTh1 --model=AR
python EvaluateModel.py --dataset=ETTh2 --model=AR
python EvaluateModel.py --dataset=ETTm1 --model=AR
python EvaluateModel.py --dataset=ETTm2 --model=AR
python EvaluateModel.py --dataset=Exchange --model=AR
python EvaluateModel.py --dataset=Weather --model=AR
python EvaluateModel.py --dataset=Illness --model=AR
python EvaluateModel.py --dataset=Divvy --model=AR
python EvaluateModel.py --dataset=M5 --model=AR
python EvaluateModel.py --dataset=Electricity --model=AR
