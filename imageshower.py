


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from import_for_notebooks import *
torch.set_default_dtype(torch.float64)

import common_utils
import analysis
import analysis_utils
from analysis import find_nearest_neighbour, scale, sort_by_metric

# in case you have your own sweep:
# sweep_id = 'uvl74ek9'
# sweeps_dir = './data/sweeps/'
# sweep = analysis_utils.read_sweep(sweeps_dir, sweep_id, name=None, problem='mnist_odd_even')
# analysis_utils.download_sweep_results_from_wandb(sweep, max_runs_to_download=100)
# X = analysis_utils.get_all_reconstruction_outputs(sweep, verbose=True)

# read sweep parameters
paths = [
    "./results/2023_03_14_18_mnist_odd_even_d250/x/49000_x.pth",
    f'./results/2023_03_14_18_mnist_odd_even_d75_mnist_odd_even/x/train'
]
sweep = common_utils.common.load_dict_to_obj("./reconstructions/mnist_odd_even/sweep.txt")
# read model, data, and whatever needed
args, Xtrn, Ytrn, ds_mean, W, model = analysis_utils.sweep_get_data_model(sweep, paths[1], put_in_sweep=True, run_train_test=True, is_federated = True)
#print(ds_mean)


# Read Reconstructed Data:


# "X" will contain a batch of all reconstructed samples (not all of them are good.. for this we need the rest of the cell)
# Here we put reconstructed data from two diffferent runs.
# you can use both of them or just one. (uncomment relevant parts)

X = torch.load(paths[0])

print(X.shape)


# Find "Good" Resonctructions (as detailed in Appendix B.3):

# Find Nearest Neighbour
print(Xtrn.shape)
xx1 = find_nearest_neighbour(X.to(torch.float32), Xtrn.to(torch.float32), search='ncc', vote='min', use_bb=False, nn_threshold=None)

# Scale to Images

xx_scaled, yy_scaled = scale(xx1, Xtrn, ds_mean)

# i = 0
# while i<len(xx_scaled):
#     #print(i)
#     j = 0
#     while j < len(yy_scaled):
#         if i==j: 
#             j+=1
#             continue
#         #print( (xxa[i]==xxa[j]).sum().item()==xxa[i].numel())
#         if (xx_scaled[i]==yy_scaled[j]).sum().item()==xx_scaled[i].numel():
#             xx_scaled = torch.cat([xx_scaled[:j], xx_scaled[j+1:]], dim = 0)
#             yy_scaled = torch.cat([yy_scaled[:j], yy_scaled[j+1:]], dim = 0)
#         else:
#             j+=1
#     i+=1
# # Sort

xxa, yya, ssims, sort_idxs = sort_by_metric(xx_scaled, yy_scaled, sort='ssim')
xxb, yyb, ssims, sort_idxs = sort_by_metric(xx_scaled, yy_scaled, sort='l2')

values = model(Xtrn).data

# Plot
# color_by_labels = Ytrn[sort_idxs]
color_by_labels = None
figpath="images/reconstructionsfed"
yya = torch.minimum(yya, torch.ones_like(xxa))
yya = torch.maximum(yya, torch.zeros_like(xxa))
yyb = torch.minimum(yyb, torch.ones_like(xxa))
yyb = torch.maximum(yyb, torch.zeros_like(xxa))




analysis.plot_table(xxa, yya, fig_elms_in_line=15, fig_lines_per_page=4, fig_type='one_above_another', color_by_labels=color_by_labels, figpath=figpath+"ssim.png", show=True, dpi=100)
analysis.plot_table(xxb, yyb, fig_elms_in_line=15, fig_lines_per_page=4, fig_type='one_above_another', color_by_labels=color_by_labels, figpath=figpath+"euclidean.png", show=True, dpi=100)