import torch
from CreateModel import CNN, NeuralNetwork
import matplotlib.pyplot as plt
def load_weights(model, fpath, device='cuda'):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath, map_location=device)

    weights['state_dict'] = {k.replace('convnet','layers'): v for k, v in weights['state_dict'].items()}

    model.load_state_dict(weights['state_dict'])
    return model

model = CNN(num_layers=6, output_dim=10)
model = load_weights(model, "results/convtestcont10l6/model_250_0_100_mnist_odd_even.pt").to(torch.device("cuda"))
model.layers = model.layers[:-1]

X, Y = torch.load("results/convtestcont10l6/x/train_0.pt")[0]
X = X.to(torch.float32)

tot_extracts = [[], [], [], [], [], [], [], [], [], []]
print(tot_extracts)
for i in range(X.shape[0]):
    extract = model(X[i].unsqueeze(0))
    
    tot_extracts[int(Y[i])].append(extract.flatten())

# tot_extracts[1]/=Y.sum()
# tot_extracts[0]/=500-Y.sum()


for cl in range(10):
    euclideans = 0
    dots = 0
    for ind1, i in enumerate(tot_extracts[cl]):
        for ind2, j in enumerate(tot_extracts[cl]):
            euclideans += torch.linalg.norm(i - j)
            dots += torch.dot(i , j)/torch.linalg.norm(i)/torch.linalg.norm(j)
    print("same", euclideans/len(tot_extracts[cl])**2, dots/len(tot_extracts[cl])**2)
    for cl2 in range(10):
        if cl == cl2: continue
        euclideans = 0
        dots = 0
        
        for ind1, i in enumerate(tot_extracts[cl]):
            for ind2, j in enumerate(tot_extracts[cl2]):
                euclideans += torch.linalg.norm(i - j)
                dots += torch.dot(i , j)/torch.linalg.norm(i)/torch.linalg.norm(j)
        print("other", euclideans/(len(tot_extracts[cl])*len(tot_extracts[cl2])), dots/(len(tot_extracts[cl])*len(tot_extracts[cl2])))
    print("++++++++++++++++++++++++++++++++++++++++++++++++")


