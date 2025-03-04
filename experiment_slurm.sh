source activate timegrad

cd /vast/s224075134/timeseries/TimeGrad-re/

for data in exchange_rate_nips electricity_nips traffic_nips solar_nips exchange_rate electricity traffic weather
do
    for seed in 2025
    do
        echo "Running timegrad for $data with seed $seed"

        sbatch  --ntasks=1 \
                --cpus-per-task=8 \
                --mem=32G \
                --gres=gpu:v100:1 \
                --qos=batch-short \
                --job-name=timegrad_${data}_${seed} \
                -o /vast/s224075134/timeseries/TimeGrad-re/stdout/timegrad_${data}_${seed}.out \
                --wrap="./scripts/timegrad.sh $data $seed"
    done
done
