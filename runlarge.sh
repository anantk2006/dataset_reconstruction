# testnum=0
# for het in 0.0 0.2 0.4 0.6 0.8 1
# do 
#     testname="results/test${testnum}2"
#     echo $testname
#     bash run.sh mnist_odd_even mnist_odd_even 75 10000000000 0.01 1000 8 8 0.0001 $het $testname
#     wait -n $!
#     u
#     CUDA_VISIBLE_DEVICES=0 \
#     python3 Main.py --data_per_class_train=75 --extraction_data_amount_per_class=500 --extraction_epochs=50000 \
#      --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 \
#      --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] \
#      --model_init_list=[0.001,0.001] --pretrained_model_path=$testname/model_75_$(($testnum * 200))_8_mnist_odd_even.pt --problem=mnist_odd_even \
#      --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=${testname}recon 
#     wait -n $!
#     echo $testname/model_75_$(($testnum * 2000))_8_mnist_odd_even.pt
#     echo ${testname}recon 
#     testnum=$((testnum+1))
    
# done


# for interval in 4 8 16 32 64 128
# do 
#     testname="results/test$testnum"
    
#     bash run.sh mnist_odd_even mnist_odd_even 250 10000000000 0.01 1000 8 $interval 0.0001 0 $testname
#     wait -n $!
    
#     CUDA_VISIBLE_DEVICES=0 \
#     python3 Main.py --data_per_class_train=250 --extraction_data_amount_per_class=500 --extraction_epochs=50000 \
#      --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 \
#      --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] \
#      --model_init_list=[0.001,0.001] --pretrained_model_path=$testname/model_250_0_${interval}_mnist_odd_even.pt --problem=mnist_odd_even \
#      --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=${testname}recon 
#     wait -n $!
     
#     testnum=$((testnum+1))
    
# done


for i in {3..6}
do 
    testname="results/convtestlayers$i"
    echo $testname
    CUDA_VISIBLE_DEVICES=0 python3 Main.py --run_mode=train --problem=mnist_odd_even --proj_name=mnist_odd_even \
    --data_per_class_train=250 --model_hidden_list=[1000,1000] --model_init_list=[0.0001,0.0001] --train_epochs=1000000000 \
    --train_lr=0.01 --train_evaluate_rate=1000 --is_federated=false --train_threshold=0.0001 --train_to_loss=true --model_type=conv \
    --output_dir=$testname --seed=4 --num_conv_layers=$i
    
#     bash run.sh mnist_odd_even mnist_odd_even 250 10000000000 0.01 1000 8 $interval 0.0001 0 $testname
    wait -n $!
    
#     
    PIDS=(0 1)
    CUDA_VISIBLE_DEVICES=0 \
    python3 Main.py --data_per_class_train=250 --extraction_data_amount_per_class=500 --extraction_epochs=50000 \
     --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 \
     --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] \
     --model_init_list=[0.001,0.001] --pretrained_model_path=$testname/model_250_0_100_mnist_odd_even.pt --problem=mnist_odd_even \
     --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=${testname}recon --model_type=conv --num_conv_layers=$i \
     --y_param=false
    
    PIDS[0]=$!

    CUDA_VISIBLE_DEVICES=1 \
    python3 Main.py --data_per_class_train=250 --extraction_data_amount_per_class=500 --extraction_epochs=50000 \
     --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 \
     --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] \
     --model_init_list=[0.001,0.001] --pretrained_model_path=$testname/model_250_0_100_mnist_odd_even.pt --problem=mnist_odd_even \
     --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=${testname}reconydiff --model_type=conv --num_conv_layers=$i \
     --y_param=true

    PIDS[1]=$!

    for pid in ${!PIDS[@]};
    do 
        wait -n $pid
        echo $pid
    done

#     testnum=$((testnum+1))
    
done


