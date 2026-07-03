#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Module load
if command -v module &> /dev/null; then
  module load gcc/15.1.0/gcc-15.1.0
  module load miniconda3/25.5.1/none-none
  cd /gpfs/workdir/fujitayo/semi-supervisedSE
else
  REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  cd "$REPO_ROOT"
fi

# Get conda environment path
if [ -z "$HOME" ]; then
  HOME=~
fi

if [ -d "$HOME/miniconda3/envs/eval_baselines" ]; then
  CONDA_ENV_PATH="$HOME/miniconda3/envs/eval_baselines"
else
  CONDA_ENV_PATH="$HOME/.conda/envs/eval_baselines"
fi

echo "Experiment Directory: $EXPDIR"
echo "Experiment Name: $EXPNAME"
echo "Configuration Name: $CFGNAME"
echo "N iterations candidates: $N_ITERS"
echo "Dataset config: $DATASET_CFGPATH"

for i in $N_ITERS; do
  if [ $i -eq 1 ]; then
    config_path_i="$EXPDIR/.hydra/$CFGNAME.yaml"
  else
    config_path_i="$EXPDIR/.hydra/${CFGNAME}_n${i}.yaml"
  fi
  exp_name_i="${EXPNAME}_N${i}"

  echo "Experiment name: $exp_name_i"
  echo "Configuration path: $config_path_i"

  # python inference.py \
  #   --exp_dir $EXPDIR \
  #   --cfg_path $config_path_i \
  #   --exp_name $exp_name_i \
  #   --dataset_cfg_path $DATASET_CFGPATH 

  # python utils/dnsmos.py --test_exp_dir $EXPDIR/$exp_name_i --cfg_path $config_path_i --eval_bound True
  # python utils/cossim.py --test_exp_dir $EXPDIR/$exp_name_i --cfg_path $config_path_i
  # python utils/dwer.py --test_exp_dir $EXPDIR/$exp_name_i --cfg_path $config_path_i --eval_bound True
  # python utils/pesq.py --test_exp_dir $EXPDIR/$exp_name_i --cfg_path $config_path_i
  # python utils/stoi.py --test_exp_dir $EXPDIR/$exp_name_i --cfg_path $config_path_i
  # python utils/sisdr.py --test_exp_dir $EXPDIR/$exp_name_i --cfg_path $config_path_i
  
  python utils/aggregate_metrics_csv.py --test_exp_dir $EXPDIR/$exp_name_i --cfg_path $config_path_i --eval_bound True
done

echo "Inference and evaluation completed!" 