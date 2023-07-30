# # for i in {0..7}
# # do
# #     python imageshower.py convfederatedmodbase${i}/x/16000_x.pth convfederated/x/train True 8tri
# # done
# COEFFS=(21 5 61 1)
# MARGINS=(750 1500 2250 3000)
# INCREMENT=0
# PIDS=(1 2 3 4 5 6 7 8)
# for coeff in "${COEFFS[@]}"
# do
#     for margin in "${MARGINS[@]}"
#     do 
#         python imageshower.py convactv01l6c${coeff}m${margin}/x/19000_x.pth convtestcont01/x/train False ${coeff}c${margin}m
# # done
        
        
#     done
    
    
# done
LRS=(1 01 1)
ARELUS=(10 50 100)
MINLS=(1 2 4)

INCREMENT=0
PIDS=(1 2 3 4 5 6 7 8)
for lr in "${LRS[@]}"
do
    for arelu in "${ARELUS[@]}"
    do 
        for minl in "${MINLS[@]}"
        do 
        
            # echo $lr $arelu $minl
            
            # CUDA_VISIBLE_DEVICES=$INCREMENT python Main.py --data_per_class_train=50 --extraction_data_amount_per_class=100 --extraction_epochs=20000 --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.$lr --extraction_min_lambda=0.$minl --extraction_model_relu_alpha=$arelu --model_hidden_list=[1000,1000] --model_init_list=[0.001,0.001] --pretrained_model_path=results/convtestcont10l6/model_50_0_100_mnist_odd_even.pt --problem=mnist_odd_even --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=results/convmultilr$(($lr))a$(($arelu))ml$(($minl)) --model_type=conv --num_conv_layers=6 --y_param=false --cont_obj=false --seed=14 --gpuid=0 --multi_class=true &
            PIDS[$INCREMENT]=$!
            python imageshower.py convmultilr${lr}a${arelu}ml${minl}/x/15000_x.pth convtestcont10l6/x/train False base
            INCREMENT=$(($INCREMENT+1))
            if [ $INCREMENT -eq 8 ]
                then
                    echo waiting
                    for pid in "${PIDS[@]}"
                    do 
                        wait -n $pid
                        echo $pid
                    done
                    PIDS=(1 2 3 4 5 6 7 8)
                    INCREMENT=0
            fi
        done
    done
    
    
done