import argparse
import ast


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    return ast.literal_eval(v)


def get_args(*args):
    parser = argparse.ArgumentParser(description='')

    # general parameters
    parser.add_argument('--cuda', default='true', type=str2bool, help='')
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--proj_name', default='', help='description of run, for saving stuff')
    parser.add_argument('--precision', default='double', help='')
    parser.add_argument('--run_mode', help='options: train, reconstruct', choices=['train', 'reconstruct'])

    # files
    # parser.add_argument('--datasets_dir', default='datasets', help='this is loaded from settings.py')
    # parser.add_argument('--results_base_dir', default='./', help='this is loaded from settings.py')
    parser.add_argument('--pretrained_model_path', default='', help='path to pretrained model (ignored if empty)')
    parser.add_argument('--output_dir', default='', help='place where results are saved')
    parser.add_argument('--train_save_model', default='true', type=str2bool, help='')
    parser.add_argument('--train_save_model_every', default=-1, type=int, help='only saves if >0')
    parser.add_argument('--extract_save_results', default='true', type=str2bool, help='')
    parser.add_argument('--extract_save_results_every', default=10000, type=int, help='only saves if >0')
    parser.add_argument('--save_args_files', default='true', type=str2bool, help='')

    # wandb
    parser.add_argument('--wandb_active', default='false', type=str2bool, help='actviate wandb logging')
    parser.add_argument('--wandb_project_name', default='Dataset_Extraction', help='')
    parser.add_argument('--wandb_entity', default='dataset_extraction', help='')

    # data creation
    parser.add_argument('--problem', help='')
    parser.add_argument('--data_per_class_train', type=int, help='')
    parser.add_argument('--data_per_class_val', default=0,type=int, help='')
    parser.add_argument('--data_per_class_test', default=1000, type=int, help='')
    parser.add_argument('--data_reduce_mean', default='true', type=str2bool, help='')
    parser.add_argument("--train_to_loss", default = "false", type = str2bool, help = "train to loss threshold strictly, or have epoch limit")

    # model_train
    parser.add_argument('--model_type', default='mlp', help='options: mlp')
    parser.add_argument('--model_hidden_list', default='[1000, 1000]', type=str2list, help='should be a list, even for one item. ')
    parser.add_argument('--model_use_bias', default='false', type=str2bool, help='')

    # train
    parser.add_argument('--use_init_scale', default='true', type=str2bool, help='')
    parser.add_argument('--use_init_scale_only_first', default='true', type=str2bool, help='')
    parser.add_argument('--model_init_list', default='[1e-01, 1e-01]', type=str2list, help='should be a list of the same size as model_hidden_list')
    parser.add_argument('--model_train_activation', default='relu', help='options: relu, sigmoid, leakyrelu, tanh')
    parser.add_argument('--train_epochs', default=1000, type=int, help='')
    parser.add_argument('--train_lr', default=0.01, type=float, help='')
    parser.add_argument('--train_evaluate_rate', default=1000, type=int, help='')
    parser.add_argument('--train_threshold', default=0.0001, type=float, help='stop training below this loss')
    parser.add_argument('--train_SGD', default='false', type=str2bool, help='')
    parser.add_argument('--train_SGD_batch_size', default=64, type=int, help='')
    parser.add_argument('--y_param', default="false", type=str2bool, help='allow making y a trainable parameter')
    parser.add_argument('--num_conv_layers', default="3", type=int, help='if conv, number of convolutional layers')
    parser.add_argument('--num_feat_extract', default="128", type=int, help='number of feature extractions for contrastive learning')

    # extraction
    parser.add_argument('--extraction_epochs', default=1000, type=int, help='')
    parser.add_argument('--extraction_data_amount_per_class', type=int, help='0 = same as data_amount')
    parser.add_argument('--extraction_model_activation', default='modifiedrelu', help='options: same as model_train_activation')
    parser.add_argument('--extraction_model_relu_alpha', default=300, type=float, help='')
    parser.add_argument('--extraction_init_scale', default=1e-9, type=float, help='')
    parser.add_argument('--extraction_lr', default=1e-4, type=float, help='')
    parser.add_argument('--extraction_lambda_lr', default=1e-4, type=float, help='')
    parser.add_argument('--extraction_lr_x', default=1e-4, type=float, help='')
    parser.add_argument('--extraction_lr_l', default=1e-4, type=float, help='')
    parser.add_argument('--extraction_evaluate_rate', default=250, type=int, help='')
    parser.add_argument('--extraction_min_lambda', default=0.05, type=float, help='minimum lambda in the extraction loss')
    parser.add_argument('--extraction_loss_type', default='kkt', help='options: kkt, naive')
    parser.add_argument('--extraction_stop_threshold', default=5000, type=int)
    parser.add_argument("--model_inversion", default = "false", type = str2bool, help = "bool whether or not to use model inversion (baseline)")
    parser.add_argument("--decoder", default = "false", type = str2bool, help = "use convolutional decoder to make images more structured")

    parser.add_argument("--gpuid", default = 0, type=int, help = "number of gpus with data")
    parser.add_argument("--is_federated", default = "false", type = str2bool)
    parser.add_argument("--num_clients", default = 8, type=int, help = "number of gpus with data")
    parser.add_argument("--init_method", default = "/home/akhande/dataset_reconstruction/", type=str, help = "folder where comms go")
    parser.add_argument("--avg_interval", default = "100", type=int, help = "iteration interval between consecutive averaging")
    parser.add_argument("--rank", default = 0, type=int, help = "rank of process here")
    parser.add_argument("--heterogeneity", default = 0, type=float, help = "percent of each clients dataset to be heterogenous")
    parser.add_argument('--cont_obj', default="false", type=str2bool, help='include novel contrastive objective')
    parser.add_argument('--cont_margin_i', default=12, type=float, help='alpha value for contrastive objective')
    parser.add_argument("--cont_coeff", default=0.1, type = float, help = "weight of contrastive loss")
    parser.add_argument('--cont_margin_ag', default=1000, type=float, help='alpha value for contrastive objective')
    parser.add_argument('--two_classes', default="false", type=str2bool, help='whether or not to use just 0s and 1s')
    parser.add_argument('--multi_class', default="false", type=str2bool, help='whether or not to do 10 different classes')    

    if not isinstance(args, list):
        args = args[0]
    args = parser.parse_args(args)

    return args


