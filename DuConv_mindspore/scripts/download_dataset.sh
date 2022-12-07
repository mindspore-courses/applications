#!/bin/bash

# download dataset file to ./data
TRAIN_URL=https://dataset-bj.cdn.bcebos.com/duconv/train.txt.gz
DEV_URL=https://dataset-bj.cdn.bcebos.com/duconv/dev.txt.gz
TEST_1_URL=https://dataset-bj.cdn.bcebos.com/duconv/test_1.txt.gz

mkdir data
cd ./data
wget --no-check-certificate ${TRAIN_URL}
wget --no-check-certificate ${DEV_URL}
wget --no-check-certificate ${TEST_1_URL}

gunzip train.txt.gz
gunzip dev.txt.gz
gunzip test_1.txt.gz
mv ./test_1.txt ./test.txt