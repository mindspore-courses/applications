#!/bin/bash
candidate_file=data/candidate.test.txt
score_file=output/score.txt
predict_file=output/predict.txt
load_checkpoint_path=save_model/match_kn_gene-3_4954.ckpt
python predict.py --task_name=match_kn_gene \
                  --max_seq_length=128 \
                  --batch_size=1 \
                  --eval_data_file_path=data/test.mindrecord \
                  --load_checkpoint_path=${load_checkpoint_path} \
                  --save_file_path=${score_file}

python src/utils/extract.py ${candidate_file} ${score_file} ${predict_file}

# step 6: if the original file has answers, you can run the following command to get result
# if the original file not has answers, you can upload the ./output/test.result.final 
# to the website(https://ai.baidu.com/broad/submission?dataset=duconv) to get the official automatic evaluation
python src/eval.py ${predict_file} > predict.log
