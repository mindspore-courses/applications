#!/bin/bash
if [ $# -ne 1 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash convert_dataset.sh [TASK_NAME]"
    echo "for example: sh scripts/convert_dataset.sh match_kn_gene"
    echo "TASK_TYPE including [match, match_kn, match_kn_gene]"
    echo "=============================================================================================================="
exit 1
fi
TASK_NAME=$1

case $TASK_NAME in
    "match")
        DICT_NAME="data/char.dict"
        ;;
    "match_kn")
        DICT_NAME="data/char.dict"
        ;;
    "match_kn_gene")
        DICT_NAME="data/gene.dict"
        ;;
    esac

python src/reader.py --task_name=${TASK_NAME} \
                     --max_seq_len=256 \
                     --vocab_path=${DICT_NAME} \
                     --input_file=data/build.train.txt \
                     --output_file=data/train.mindrecord
python src/reader.py --task_name=${TASK_NAME} \
                     --max_seq_len=256 \
                     --vocab_path=${DICT_NAME} \
                     --input_file=data/build.dev.txt \
                     --output_file=data/dev.mindrecord
python src/reader.py --task_name=${TASK_NAME} \
                     --max_seq_len=256 \
                     --vocab_path=${DICT_NAME} \
                     --input_file=data/build.test.txt \
                     --output_file=data/test.mindrecord

