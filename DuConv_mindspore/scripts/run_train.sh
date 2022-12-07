#!/bin/bash
if [ $# -ne 1 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "sh build_dataset.sh [TASK_NAME]"
    echo "for example: sh scripts/build_dataset.sh match_kn_gene"
    echo "TASK_TYPE including [match, match_kn, match_kn_gene]"
    echo "=============================================================================================================="
exit 1
fi
TASK_NAME=$1

python train.py --epoch=30 \
                --task_name=${TASK_NAME} \
                --max_seq_length=256 \
                --batch_size=128 \
                --train_data_file_path=data/train.mindrecord \
                --save_checkpoint_path=save_model/ > train.log &