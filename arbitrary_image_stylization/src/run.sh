RANK_SIZE=8

EXEC_PATH=$(pwd)

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./train.py ./device$i
    cp -r ./model ./dataset ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ./train.py --parallel=1 > train.log$i 2>&1 &
    mkdir result
    cd ../
done