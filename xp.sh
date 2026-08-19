#!/bin/bash -l

### Set variables #########################################################
var_gres=gpu:1
var_time=1-18:00:00
var_mem=32G
var_p=gpu
var_exclude=erc-hpc-comp035,erc-hpc-comp040
var_log_dir=/scratch/prj/ccc_vit_finetuning/logs
###########################################################################


EXPERIMENTS=(q1_weight_e0 q1_weight_e1 q1_weight_e2 q1_weight_e10 q1_weight_e100)

RUNS=(run_1 run_2)

BASE_LOG_DIR=/scratch/prj/ccc_vit_finetuning/logs
UNIX_TS=$(date +%s)

for EXP_NAME in "${EXPERIMENTS[@]}"; do

    echo "=============================="
    echo "Experiment: $EXP_NAME"
    echo "=============================="


    for RUN_ID in "${RUNS[@]}"; do

        echo "Submitting: $EXP_NAME / $RUN_ID"

        sbatch \
          -p "$var_p" \
          --exclude="${var_exclude}" \
          --job-name="${EXP_NAME}_${RUN_ID}" \
          --gres="${var_gres}" \
          --time="${var_time}" \
          --mem="${var_mem}" \
          xp_launch.sh ${EXP_NAME} ${RUN_ID} ${UNIX_TS}

          sleep 3

    done
done


