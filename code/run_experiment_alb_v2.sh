#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tf26-gpu
server_id=$1
exp=$2
dataset=$3
modeltype=$4
modelsize=$5
combid=$6
epochs=$7
archNN=$8
reescritura=$9

echo "Running on server_id: $server_id"
echo "exp $exp, dataset $dataset, modeltype $modeltype, modelsize $modelsize, combid $combid, epochs $epochs, archNN $archNN, reescritura $reescritura"

if [[ $server_id = "go1" ]]; then
    echo "Setting CUDA_VISIBLE_DEVICES to 0"
    CUDA_VISIBLE_DEVICES=0
elif [[ $server_id == "go2" ]]; then
    echo "Setting CUDA_VISIBLE_DEVICES to 1"
    CUDA_VISIBLE_DEVICES=1
else
    echo "Setting CUDA_VISIBLE_DEVICES to 0"
    CUDA_VISIBLE_DEVICES=0
fi

echo "Running experiment: $exp - DATASET: $dataset - MODEL_TYPE: $modeltype - MODEL_SIZE: $modelsize - COMB_ID: $combid - EPOCHS: $epochs"

fich="EXP_${exp}_${dataset}_${modeltype}_${modelsize}_${combid}_output.log" 
fich2="EXP_${exp}_${dataset}_${modeltype}_${modelsize}_${combid}_error.log" 
dir="EXP_${exp}_${dataset}_${modeltype}_${modelsize}" 

id_file=$RANDOM

mkdir -p ${dataset}_output_logs
mkdir -p ${dataset}_output_logs/old
mkdir -p ${dataset}_output_logs/borrar
mkdir -p ${dataset}_output_logs/${dir}

mkdir -p ${dataset}_output
mkdir -p ${dataset}_output/old
mkdir -p ${dataset}_output/borrar
mkdir -p ${dataset}_output/${dir}

#plots_hists_debug/EXP_0_crypto_NN_large/2
mkdir -p plots_hists_debug
mkdir -p plots_hists_debug/old
mkdir -p plots_hists_debug/borrar
mkdir -p plots_hists_debug/${dir}

if [ ${reescritura}n == "n" ]
then
	echo "Reescritura NO. Moviendo a old"
	mv "./${dataset}_output_logs/${dir}/${fich}" "./${dataset}_output_logs/old/${fich}_${id_file}"
	mv "./${dataset}_output_logs/${dir}/${fich2}" "./${dataset}_output_logs/old/${fich2}_${id_file}"
	mv "./${dataset}_output/${dir}/${combid}" "./${dataset}_output_logs/old/${dir}_${combid}_${id_file}"
	mv "./plots_hists_debug/${dir}/${combid}" "./plots_hists_debug/old/${dir}_${combid}_${id_file}"
else
	echo "Reescritura directorio.. Moviendo a borrar"
	mv "./${dataset}_output_logs/${dir}/${fich}" "./${dataset}_output_logs/borrar/${fich}_${id_file}"
	mv "./${dataset}_output_logs/${dir}/${fich2}" "./${dataset}_output_logs/borrar/${fich2}_${id_file}"
	mv "./${dataset}_output/${dir}/${combid}" "./${dataset}_output_logs/borrar/${dir}_${combid}_${id_file}"
	mv "./plots_hists_debug/${dir}/${combid}" "./plots_hists_debug/borrar/${dir}_${combid}_${id_file}"
fi

echo "Ejecutar .... "

python AdvGAN-FINAL_alb.py --exp $exp --dataset $dataset --modeltype $modeltype --modelsize $modelsize --combid $combid --epochs $epochs > >(tee ./${dataset}_output_logs/EXP_${exp}_${dataset}_${modeltype}_${modelsize}/EXP_${exp}_${dataset}_${modeltype}_${modelsize}_${combid}_output.log) 2> >(tee ./${dataset}_output_logs/EXP_${exp}_${dataset}_${modeltype}_${modelsize}/EXP_${exp}_${dataset}_${modeltype}_${modelsize}_${combid}_error.log >&2)

#python AdvGAN-FINAL_alb.py --exp $exp --dataset $dataset --modeltype $modeltype --modelsize $modelsize --combid $combid --epochs $epochs --archNN "$archNN" 2>&1  | tee ./${dataset}_output_logs/EXP_${exp}_${dataset}_${modeltype}_${modelsize}/EXP_${exp}_${dataset}_${modeltype}_${modelsize}_${combid}_output.log 
