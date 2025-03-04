cd /vast/s224075134/timeseries/TimeGrad-re/

dataset=$1
seed=$2

for pred_len in 48 96 192 336 720
do
  python timegrad_training.py --dataset ${dataset} --prediction_length ${pred_len} --seed $seed
done
