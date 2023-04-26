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
OUTPUT=${11}






SHAREDFILE="file:///home/akhande/dataset_reconstruction/federatedfiles/sharedfile.pt"



PIDS=(1 2 3 4 5 6 7 8)
# placeholder values
for i in {0..7}
do
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
        --output_dir=$OUTPUT \
        --model_type=conv \
        --num_conv_layers=6 \
        --cont_obj=false \
        --y_param=false \
        --seed=$i &
    PIDS[$i]=$!
    

    #pids="${pids} $!"
    
done

for pid in ${!PIDS[@]};
do 
    wait -n $pid
    echo $pid
done


#echo "children:${pids}"
#wait
