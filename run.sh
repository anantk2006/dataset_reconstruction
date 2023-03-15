#!/bin/bash



PROBLEM=$1
PROJ_NAME=$2
DATA_PER_CLASS_TRAIN=$3
TRAIN_EPOCHS=$4
TRAIN_LR=$5
TRAIN_EVALUATE_RATE=$6

WORLD_SIZE=$7
AVG_INTERVAl=$8
THRES=$9
HETERO=${10}




SHAREDFILE="file:///home/akhande/dataset_reconstruction/federatedfiles/sharedfile.pt"



pids=""
for i in {0..7}
do
    #echo dddddddddddddddddddddddddddddddddddddddddddddd
    # if [ $i -eq 0 ]
    # then
    #     continue
    # fi
    # if [ $i -eq 4 ]
    # then
    #     continue
    # fi
    RANK=$i
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python Main.py \
        --run_mode=train \
        --problem=$PROBLEM \
        --proj_name=$PROJ_NAME \
        --data_per_class_train=$DATA_PER_CLASS_TRAIN \
        --model_hidden_list=[1000,1000] \
        --model_init_list=[0.0001,0.0001] \
        --train_epochs=$TRAIN_EPOCHS \
        --train_lr=$TRAIN_LR \
        --train_evaluate_rate=$TRAIN_EVALUATE_RATE \
        --train_to_loss=true \
        --train_threshold=$THRES \
        --rank=$RANK \
        --num_clients=$WORLD_SIZE \
        --init_method=$SHAREDFILE \
        --avg_interval=$AVG_INTERVAl \
        --heterogeneity=$HETERO \
        --seed=$i &
    

    #pids="${pids} $!"
    
done

#echo "children:${pids}"
#wait