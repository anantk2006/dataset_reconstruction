# PIDS=(0 1 2 3 4 5 6 7)
# for i in {0..7}
# do
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Main.py --data_per_class_train=250 --extraction_data_amount_per_class=500 --extraction_epochs=20000 --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] --model_init_list=[0.001,0.001] --pretrained_model_path=results/convfederated/model_50_0_8_mnist_odd_even.pt --problem=mnist_odd_even --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=results/convfederatedbase${i} --model_type=conv --num_conv_layers=3 --y_param=false --cont_obj=false --seed=$(($i+10)) --gpuid=$i &
#     PIDS[$i]=$!
    
# done
# for pid in ${!PIDS[@]};
# do 
#         wait -n $pid
#         echo $pid
# done
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Main.py --data_per_class_train=250 --extraction_data_amount_per_class=500 --extraction_epochs=20000 --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] --model_init_list=[0.001,0.001] --pretrained_model_path=results/convfederated/model_50_0_8_mnist_odd_even.pt --problem=mnist_odd_even --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=results/convfederatedmodactv${i}t2 --model_type=conv --num_conv_layers=3 --y_param=false --cont_obj=true --cont_margin_ag=800 --cont_margin_i=800 --cont_coeff=0.002 --seed=$(($i+10)) --gpuid=$i &
    PIDS[$i]=$!
    
done
for pid in ${!PIDS[@]};
do 
        wait -n $pid
        echo $pid
done
# for i in {0..7}
# do
#     CUDA_VISIBLE_DEVICES=$i python Main.py --data_per_ class_train=250 --extraction_data_amount_per_class=500 --extraction_epochs=20000 --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] --model_init_list=[0.001,0.001] --pretrained_model_path=results/convfederated/model_50_0_8_mnist_odd_even.pt --problem=mnist_odd_even --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=results/convfederatedmodbase${i} --model_type=conv --num_conv_layers=3 --y_param=false --cont_obj=false --seed=$(($i+10)) &
#     PIDS[$i]=$!
    
# done
# for pid in ${!PIDS[@]};
# do 
#         wait -n $pid
#         echo $pid
# done
# for i in {0..7}
# do
#     CUDA_VISIBLE_DEVICES=$i python Main.py --data_per_class_train=250 --extraction_data_amount_per_class=500 --extraction_epochs=20000 --extraction_evaluate_rate=1000 --extraction_init_scale=0.03497673778414215 --extraction_lr=0.03052419903283405 --extraction_min_lambda=0.4470505589528116 --extraction_model_relu_alpha=149.86555429083975 --model_hidden_list=[1000,1000] --model_init_list=[0.001,0.001] --pretrained_model_path=results/convtestcontl6/model_250_0_100_mnist_odd_even.pt --problem=mnist_odd_even --run_mode=reconstruct --wandb_active=False --is_federated=false --output_dir=results/convtestcontrecon3actvl6${i} --model_type=conv --num_conv_layers=6 --y_param=false --cont_obj=true --cont_margin=10 --cont_coeff=0.001 --seed=$(($i+10)) &
#     PIDS[$i]=$!
    
# done
# for pid in ${!PIDS[@]};
# do 
#         wait -n $pid
#         echo $pid
# done