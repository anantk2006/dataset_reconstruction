import os
import sys

import threadpoolctl
import torch
import numpy as np
import datetime
import wandb

import common_utils
from common_utils.common import AverageValueMeter, load_weights, now, save_weights
from CreateData import setup_problem
from CreateModel import create_model
from extraction import calc_extraction_loss, evaluate_extraction, get_trainable_params
from GetParams import get_args

import torch.distributed as dist

thread_limit = threadpoolctl.threadpool_limits(limits=8)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

###############################################################################
#                               Train                                         #
###############################################################################
def get_loss_ce(args, model, x, y):
    p = model(x)
    p = p.view(-1)
    
    loss = torch.nn.BCEWithLogitsLoss()(p, y.to(torch.float32))
    return loss, p


def get_total_err(args, p, y):
    # BCEWithLogitsLoss needs 0,1
    err = (p.sign().view(-1).add(1).div(2) != y).float().mean().item()
    return err


# def epoch_ce_sgd(args, dataloader, model, epoch, device, batch_size, opt=None):
#     total_loss, total_err = AverageValueMeter(), AverageValueMeter()
#     model.train()
#     for i, (x, y) in enumerate(dataloader):
#         idx = torch.randperm(len(x))
#         x, y = x[idx], y[idx]
#         x, y = x.to(device), y.to(device)
#         for batch_idx in range(len(x) // batch_size + 1):
#             batch_x, batch_y = x[batch_idx * batch_size: (batch_idx + 1) * batch_size], y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
#             if len(batch_x) == 0:
#                 continue
#             loss, p = get_loss_ce(args, model, x, y)
#
#             if opt:
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()
#
#             err = get_total_err(args, p, y)
#             total_err.update(err)
#
#             total_loss.update(loss.item())
#     return total_err.avg, total_loss.avg, p.data


def epoch_ce(args, dataloader, model, epoch, device, opt=None):
    total_loss, total_err = AverageValueMeter(), AverageValueMeter()
    model.train()
    model = model.cuda()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        loss, p = get_loss_ce(args, model, x, y)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        err = get_total_err(args, p, y)
        total_err.update(err)

        total_loss.update(loss.item())
    return total_err.avg, total_loss.avg, p.data

def average_model(world_size, model):
    for param in model.parameters():
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
        param.data /= world_size
        dist.broadcast(param.data, src=0)

def train(args, train_loader, test_loader, val_loader, model):
    
    if args.is_federated:
        sys.stdout = sys.__stdout__
        print(args.rank)
        print(args.device, args.num_clients)
    # prefix = "file://"
    # assert args.init_method.startswith(prefix)
    # sharedfile_path = args.init_method[len(prefix):]
    # if os.path.isfile(sharedfile_path) and args.rank == 0:
    #     print(
    #         "\n========================\n"
    #         f"Sharedfile {sharedfile_path} already exists. Deleting now.\n"
    #         "We are assuming that this is an old sharedfile which is fine to delete."
    #         " If this sharefile is actively being used during another training run,"
    #         " then you have mistakenly provided the same sharefile path to two"
    #         " different runs. In this case, deleting the sharedfile (as we are doing"
    #         " now) will likely destroy the other training run."
    #         " Be careful to avoid this.\n"
    #         "========================\n"
    #     )
    #     os.remove(sharedfile_path)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.train_lr)
    print('Model:')
    print(model)
    print(train_loader[0][0].shape, train_loader[0][1].shape, len(train_loader))
    print(test_loader[0][0].shape, test_loader[0][1].shape, len(test_loader))

    # Handle Reduce Mean
    if args.data_reduce_mean:

        # TODO: Revert back to previous version.
        print('Reducing Trainset-Mean from Trainset and Testset')
        Xtrn, Ytrn = next(iter(train_loader))
        ds_mean = Xtrn.mean(dim=0, keepdims=True)
        Xtrn = Xtrn - ds_mean
        train_loader = [(Xtrn, Ytrn)]

        Xtst, Ytst = next(iter(test_loader))
        Xtst = Xtst - ds_mean
        test_loader = [(Xtst, Ytst)]
        
    iters = 0
    t_total = 0
    print(next(iter(train_loader))[0].shape, next(iter(test_loader))[0].shape)
    torch.set_printoptions(precision=20)
    for epoch in range(args.train_epochs + 1):
        # if args.train_SGD:
        #     train_error, train_loss, output = epoch_ce_sgd(args, train_loader, model, epoch, args.device, args.train_SGD_batch_size, optimizer)
        # else:
        total_loss, total_err = AverageValueMeter(), AverageValueMeter()
        model.train()
        for i, (x, y) in enumerate(train_loader):
            
            if args.is_federated and iters > args.avg_interval:
                
                dist.barrier()
                average_model(args.num_clients, model)
                
                iters = 0
            t_total+= y.numel()
            x, y, model = x.cuda(), y.cuda(), model.cuda()
            loss, p = get_loss_ce(args, model, x, y)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            err = get_total_err(args, p, y)
            total_err.update(err)

            total_loss.update(loss.item())
            iters += 1
            
        train_loss = total_loss.avg
        train_error = total_err.avg
        output = p.data

        train_loss = torch.Tensor([train_loss]).cuda()
        if args.is_federated:
            
            dist.barrier()
            
            dist.reduce(train_loss, dst=0, op=dist.ReduceOp.SUM)
            train_loss /= args.num_clients
            dist.broadcast(train_loss, src=0)

        if train_loss<args.train_threshold:
            if args.is_federated:
                
                dist.barrier()

                average_model(args.num_clients, model)
            break
        
        
            

        train_l = float(train_loss.item())
        if epoch % args.train_evaluate_rate == 0:
            test_error, test_loss, _ = epoch_ce(args, test_loader, model, args.device, None, None)
            if val_loader is not None:
                validation_error, validation_loss, _ = epoch_ce(args, val_loader, model, args.device, None, None)
                
                print(now(), f'Epoch {epoch}: train-loss = {train_l:.8g} ; train-error = {train_error:.4g} ; test-loss = {test_loss:.8g} ; test-error = {test_error:.4g} ; validation-loss = {validation_loss:.8g} ; validation-error = {validation_error:.4g} ; p-std = {output.abs().std()}; p-val = {output.abs().mean()}')
            else:
                print(now(),
                      f'Epoch {epoch}: train-loss = {train_l:.8g} ; train-error = {train_error:.4g} ; test-loss = {test_loss:.8g} ; test-error = {test_error:.4g} ; p-std = {output.abs().std()}; p-val = {output.abs().mean()}')

            if args.wandb_active:
                wandb.log({"epoch": epoch, "train loss": train_loss, 'train error': train_error, 'p-val':output.abs().mean(), 'p-std': output.abs().std()})
                if val_loader is not None:
                    wandb.log({'validation loss': validation_loss, 'validation error': validation_error})
                wandb.log({'test loss': test_loss, 'test error': test_error})

        if np.isnan(train_l):
            raise ValueError('Optimizer diverged')

        

        if args.rank ==0 and args.train_save_model_every > 0 and epoch % args.train_save_model_every == 0:
            save_weights(os.path.join(args.output_dir, 'weights'), model, ext_text=args.model_name, epoch=epoch)
        

    print(now(), 'ENDED TRAINING')
    if args.is_federated: 
        print("final agg")
        dist.barrier()
        average_model(args.num_clients, model)
    print(f"{args.rank} made it to end")
            
    return model, train_loader


###############################################################################
#                               Extraction                                    #
###############################################################################

def data_extraction(args, dataset_loader, model):

    # we use dataset only for shapes and post-visualization (adding mean if it was reduced)
    x0, y0 = next(iter(dataset_loader))
    x0, y0 = x0.to(model.layers[0].weight.device), y0.to(model.layers[0].weight.device)
    print('X:', x0.shape, x0.device)
    print('y:', y0.shape, y0.device)
    print('model device:', model.layers[0].weight.device)
    if args.data_reduce_mean:
        ds_mean = x0.mean(dim=0, keepdims=True)
        x0 = x0 - ds_mean

    # # send inputs to wandb/notebook
    # if args.wandb_active:
    #     send_input_data(args, model, x0, y0)

    # create labels (equal number of 1/-1)
    y = torch.zeros(args.extraction_data_amount).type(torch.get_default_dtype()).to(args.device)
    y[:y.shape[0] // 2] = -1
    y[y.shape[0] // 2:] = 1
    y = y.long()

    # trainable parameters
    l, opt_l, opt_x, x = get_trainable_params(args, x0)

    print('y type,shape:', y.type(), y.shape)
    print('l type,shape:', l.type(), l.shape)

    torch.save(y, os.path.join(args.output_dir, "y.pth"))
    if args.wandb_active:
        wandb.save(os.path.join(wandb.run.dir, "y.pth"), base_path=args.wandb_base_path)

    # extraction phase
    for epoch in range(args.extraction_epochs):
        values = model(x).squeeze()
        loss, kkt_loss, loss_verify = calc_extraction_loss(args, l, model, values, x, y)
        if np.isnan(kkt_loss.item()):
            raise ValueError('Optimizer diverged during extraction')
        opt_x.zero_grad()
        opt_l.zero_grad()
        loss.backward()
        opt_x.step()
        opt_l.step()

        if epoch % args.extraction_evaluate_rate == 0:
            extraction_score = evaluate_extraction(args, epoch, kkt_loss, loss_verify, x, x0, y0, ds_mean)
            if epoch >= args.extraction_stop_threshold and extraction_score > 3300:
                print('Extraction Score is too low. Epoch:', epoch, 'Score:', extraction_score)
                break

        # send extraction output to wandb
        if (args.extract_save_results_every > 0 and epoch % args.extract_save_results_every == 0) \
                or (args.extract_save_results and epoch % args.extraction_evaluate_rate == 0):
            torch.save(x, os.path.join(args.output_dir, 'x', f'{epoch}_x.pth'))
            torch.save(l, os.path.join(args.output_dir, 'l', f'{epoch}_l.pth'))
            if args.wandb_active:
                wandb.save(os.path.join(args.output_dir, 'x', f'{epoch}_x.pth'), base_path=args.wandb_base_path)
                wandb.save(os.path.join(args.output_dir, 'l', f'{epoch}_l.pth'), base_path=args.wandb_base_path)


###############################################################################
#                               MAIN                                          #
###############################################################################
def create_dirs_save_files(args):
    if args.rank==0:
        if args.train_save_model or args.extract_save_results or args.extract_save_results:
            # create dirs
            os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, 'x'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, 'l'), exist_ok=True)

        if args.save_args_files:
            # save args
            common_utils.common.dump_obj_with_dict(args, f"{args.output_dir}/args.txt")
            # save command line
            with open(f"{args.output_dir}/sys.args.txt", 'w') as f:
                f.write(" ".join(sys.argv))


def setup_args(args):
    torch.manual_seed(args.seed)
    from settings import datasets_dir, models_dir, results_base_dir
    args.datasets_dir = datasets_dir
        
    args.device = torch.device(f'cuda:{args.rank}' if torch.cuda.is_available() else 'cpu')
    args.results_base_dir = results_base_dir
    if args.pretrained_model_path:
        args.pretrained_model_path = os.path.join(models_dir, args.pretrained_model_path)
    args.model_name = f'{args.problem}_d{args.data_per_class_train}'
    if args.proj_name:
        args.model_name += f'_{args.proj_name}'
    
    if args.rank == 0:
        

        
        

    

        if args.wandb_active:
            wandb.init(project=args.wandb_project_name, entity='dataset_reconsruction')
            wandb.config.update(args)

        if args.wandb_active:
            args.output_dir = wandb.run.dir
        args.wandb_base_path = './'
    import dateutil.tz
    timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M')
    run_name = f'{timestamp}_{args.model_name}'
    args.output_dir = os.path.join(args.results_base_dir, run_name)
    print('OUTPUT_DIR:', args.output_dir)
    
    return args


def main_train(args, train_loader, test_loader, val_loader):
    print('TRAINING A MODEL')
    model = create_model(args, extraction=False)
    if args.wandb_active:
        wandb.watch(model)
    
    
    trained_model, train_loader = train(args, train_loader, test_loader, val_loader, model)
    if args.rank==0 and args.train_save_model:
        save_weights(args.output_dir, trained_model, ext_text=args.model_name)
    for i in range(args.num_clients):
        if args.rank == i:
            torch.save(train_loader, args.output_dir + f"/x/train_{args.rank}.pt")

def main_reconstruct(args, train_loader):
    print('USING PRETRAINED MODEL AT:', args.pretrained_model_path)
    extraction_model = create_model(args, extraction=True)
    extraction_model.eval()
    extraction_model = load_weights(extraction_model, args.pretrained_model_path, device=args.device)
    print('EXTRACTION MODEL:')
    print(extraction_model)

    data_extraction(args, train_loader, extraction_model)


def validate_settings_exists():
    if os.path.isfile("settings.py"):
        return
    raise FileNotFoundError("You should create a 'settings.py' file with the contents of 'settings.deafult.py', " + 
                            "adjusted according to your system")


def main():
    
    print(now(), 'STARTING!')
    validate_settings_exists()
    args = get_args(sys.argv[1:])
    args = setup_args(args)
    if args.rank != 0:
        dn = open(os.devnull, "w")
        sys.stdout = dn
    create_dirs_save_files(args)
    print('ARGS:')
    print(args)
    print('*'*100)
    if args.is_federated: 
        print("Moment of truth")
        dist.init_process_group(backend="nccl", rank=args.rank, world_size=args.num_clients, init_method=args.init_method)
        print("Hallelujah")
    
    if args.precision == 'double':
        torch.set_default_dtype(torch.float64)
    if args.cuda:
        print(f'os.environ["CUDA_VISIBLE_DEVICES"]={os.environ["CUDA_VISIBLE_DEVICES"]}')
    torch.cuda.set_device(args.rank)

    print('DEVICE:', args.device)
    print('DEFAULT DTYPE:', torch.get_default_dtype())

    train_loader, test_loader, val_loader = setup_problem(args)
    

    # train
    if args.run_mode == 'train':
        main_train(args, train_loader, test_loader, val_loader)
    # reconstruct
    elif args.run_mode == 'reconstruct':
        main_reconstruct(args, train_loader)
    else:
        raise ValueError(f'no such args.run_mode={args.run_mode}')


if __name__ == '__main__':
    main()
