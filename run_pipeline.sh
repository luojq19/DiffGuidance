#!/bin/bash
classifier_scale=$1
guide_timestep=$2
classifier_path=$3
device=$4
noise=$5
tag=$6
mode=$7

source /nethome/jluo373/anaconda3/bin/activate targetdiff
python run_sample.py $classifier_scale $guide_timestep $classifier_path $device $noise $tag $mode

# run retro*
source /nethome/jluo373/anaconda3/bin/activate retro_star_env
result_dir=/work/jiaqi/targetdiff/results_guided/outputs_guided_crsd_${classifier_scale}_${guide_timestep}_noise${noise}_${tag}_${mode}
cd /work/jiaqi/retro_star
python process.py --input ${result_dir}/eval_results/smiles.smi

echo "classifier_scale: ${classifier_scale}"
echo "guide_timestep: ${guide_timestep}"
echo "classifier_path: ${classifier_path}"
echo "noise: ${noise}"
echo "tag: ${tag}"
echo "result_dir: ${result_dir}"