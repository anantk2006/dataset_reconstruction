COEFFS=(0025 005 0075 01)
MARGINS=(750 1500 2250 3000)
INCREMENT=0
PIDS=(1 2 3 4 5 6 7 8)
for coeff in "${COEFFS[@]}"
do
    for margin in "${MARGINS[@]}"
    do 
        
        
        CUDA_VISIBLE_DEVICES=$INCREMENT python Main.py --data_per_class_train=250 --extraction_data_amount_per_class=500 --extraction_epochs=20000 --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] --model_init_list=[0.001,0.001] --pretrained_model_path=results/convtestcont01l6/model_250_0_100_mnist_odd_even.pt --problem=mnist_odd_even --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=results/convactv01l6c$(($coeff))m$(($margin)) --model_type=conv --num_conv_layers=6 --y_param=false --cont_obj=true --cont_margin_ag=$margin --cont_margin_i=$margin --cont_coeff=0.$coeff --seed=14 --gpuid=0 --two_classes=true &
        PIDS[$i]=$!
        INCREMENT=$(($INCREMENT+1))
    done
    
    if [ $INCREMENT -eq 8 ]
    then
        echo waiting
        for pid in "${PIDS[@]}"
        do 
            wait -n $pid
            echo $pid
        done
        INCREMENT=0
    fi
done