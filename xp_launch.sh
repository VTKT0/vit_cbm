#!/bin/bash -l
#SBATCH --output=/scratch/prj/ccc_vit_finetuning/logs/%j.out     # stdout
#SBATCH --error=/scratch/prj/ccc_vit_finetuning/logs/%j.err        # stderr

EXP_NAME="$1"
RUN_ID="$2"
UNIX_TS="$3"
start_time=$(date +%s)

echo "Experiment name: $EXP_NAME"
echo "Start time: $start_time"


echo "===== JOB START ====="
echo "Running on node: $HOSTNAME"
echo "pwd = $PWD"
echo "Job ID: $SLURM_JOB_ID"

# GPU info
module load cuda
source /scratch/prj/ccc_vit_finetuning/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# create folder to store in
save_dir="${EXP_NAME}_${UNIX_TS}/${RUN_ID}"
mkdir -p "/scratch/prj/ccc_vit_finetuning/vit_cbm/results/${save_dir}"

python3 -m vit_cbm.test --config ${EXP_NAME}.yaml --run_name ${save_dir}

# Done
echo "Run complete"

end_time=$(date +%s)

echo "Total runtime: $((end_time - start_time)) seconds"

# sbatch -p gpu --exclude=erc-hpc-comp035,erc-hpc-comp040 baseline.sh