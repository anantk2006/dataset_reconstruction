import torch
import torch.nn as nn
from torch.autograd import Function

import common_utils
from resnet import ResNet, BasicBlock

def get_activation(activation, model_relu_alpha):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'modifiedrelu':
        return ModifiedRelu(model_relu_alpha)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ModifiedReluFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.set_materialize_grads(False)
        ctx.x = x
        ctx.alpha = alpha
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        return grad_output * ctx.x.mul(ctx.alpha).sigmoid(), None


class ModifiedRelu(nn.Module):
    def __init__(self, alpha):
        super(ModifiedRelu, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return ModifiedReluFunc.apply(x, self.alpha) # here you call the function!


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim, activation, use_bias=False, extracts = 128):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim_list[0])])
        for i in range(1, len(hidden_dim_list)):
            self.layers.append(nn.Linear(hidden_dim_list[i-1], hidden_dim_list[i], bias=use_bias))
        self.layers.append(nn.Linear(hidden_dim_list[-1], extracts, bias=False))  # output layer
        self.layers.append(nn.Linear(extracts, output_dim, bias=False))  # output layer

    def forward(self, data, extract = False):
        feats = Flatten()(data)
        for layer in self.layers[:-1]:
            feats = layer(feats)
            feats = self.activation(feats)
        extraction = feats
        feats = self.layers[-1](feats)
        if extract: return feats, extraction
        else: return feats
class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        

        

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 3, 
            stride=2, output_padding=0),
            
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        
        x = x.reshape(-1, 4, 3, 3)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
class CNN(nn.Module):
    def __init__(self, num_layers = 3, problem = "mnist_odd_even", extracts = 128, activation = nn.ReLU(), output_dim = 1):
        super(CNN, self).__init__()
        if problem == "mnist_odd_even":
            layer1 = nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=7,              
                stride=1,                   
                padding=0, 
                bias = False                
            )
        elif problem == "cifar10_vehicles_animals":
            layer1 = nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=11,              
                stride=1,                   
                padding=0, 
                bias = False
            )
        else:
            raise NotImplementedError(f"problem name {problem} doesn't exist")
        self.layers = nn.ModuleList([         
            layer1,                              
            activation,
            nn.Conv2d(16, 32, 6, 2, 1, bias = False),
            activation, ]                        
                                  
        )
        for i in range(num_layers-3):
            self.layers.append(nn.Conv2d(32, 32, 3, 1, 1, bias = False))
            self.layers.append(activation)
        for layer in [nn.Conv2d(32, 32, 4, 1, 0, bias = False),
            activation,
            nn.Flatten(),
            nn.Linear(32*7*7, extracts, bias = False),
            activation,
            nn.Linear(extracts, output_dim, bias = False) ]:
            self.layers.append(layer)
        self.layers = nn.Sequential(*self.layers)
        # fully connected layer, output 10 classes
     
    def forward(self, x, extract = False):
        extraction = self.layers[:-1](x)
        values = self.layers[-1:](extraction)
        if extract: return values, extraction
        else: return values

def create_model(args, extraction):
    if not extraction:
        activation = get_activation(args.model_train_activation, args.extraction_model_relu_alpha)
    else:
        activation = get_activation(args.extraction_model_activation, args.extraction_model_relu_alpha)
    num_classes = 10 if args.multi_class else 1
    if args.model_type == 'mlp':
        model = NeuralNetwork(
            input_dim=args.input_dim, hidden_dim_list=args.model_hidden_list, output_dim=num_classes,
            activation=activation, use_bias=args.model_use_bias, extracts = args.num_feat_extract
        )
    elif args.model_type == "conv":
         model = CNN(num_layers=args.num_conv_layers, problem=args.problem, extracts = args.num_feat_extract, activation = activation, output_dim=num_classes)
    elif args.model_type == "res":
        model = ResNet(img_channels=1, num_layers=18, block = BasicBlock, num_classes=num_classes, activation=activation)
    else:
        raise ValueError(f'No such args.model_type={args.model_type}')

    model = model.to(args.device)

    # initialize
    if args.use_init_scale and not extraction and args.model_type!="res":

        if not args.use_init_scale_only_first:
            assert len(args.model_init_list) == 1 + len(args.model_hidden_list), "use_init_scale_only_first=False but you didn't specify suitable model_init_list"

        # intialize bias of first layer
        if hasattr(model.layers[0], 'bias') and model.layers[0].bias is not None:
            model.layers[0].bias.data.normal_().mul_(args.model_init_list[0])

        if args.use_init_scale_only_first:
            print('Initializing model weights - Only First Layer')
            model.layers[0].weight.data.normal_().mul_(args.model_init_list[0])
        else:
            print('Initializing model weights - All Layers')
            j = 0
            for i in range(len(model.layers)):
                name = model.layers[i].__class__.__name__.lower()
                if 'conv' in name or 'linear' in name:
                    model.layers[i].weight.data.normal_().mul_(args.model_init_list[j])
                    print(i, name, 'scale', args.model_init_list[j])
                    j += 1
    elif args.model_type=="res":
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                

        model.apply(weights_init)
    else:
        print('NO INITIALIZATION OF WEIGHTS')

    common_utils.common.calc_model_parameters(model)

    return model

