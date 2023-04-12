import torch
from CreateModel import CNN, NeuralNetwork
import matplotlib.pyplot as plt
def load_weights(model, fpath, device='cuda'):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath, map_location=device)

    weights['state_dict'] = {k.replace('convnet','layers'): v for k, v in weights['state_dict'].items()}

    model.load_state_dict(weights['state_dict'])
    return model

model = CNN()
model = load_weights(model, "results/convtestcont/model_250_0_100_mnist_odd_even.pt").to(torch.device("cuda"))
model.layers = model.layers[:-1]

X, Y = torch.load("results/convtestcont/x/train_0.pt")[0]
X = X.to(torch.float32)

tot_extracts = [[], []]
for i in range(X.shape[0]):
    extract = model(X[i].unsqueeze(0))
    
    tot_extracts[int(Y[i])] += [extract.flatten()]

# tot_extracts[1]/=Y.sum()
# tot_extracts[0]/=500-Y.sum()

euclideans = 0
dots = 0
print(len(tot_extracts[1]), len(tot_extracts[0]))
for i in tot_extracts[1]:
    for j in tot_extracts[1]:
        euclideans += (i - j).abs().sum()
        dots += torch.dot(i , j)/torch.linalg.norm(i)/torch.linalg.norm(j)
print(euclideans, dots)

plt.scatter([i[26].cpu().detach() for i in tot_extracts[0]], [i[52].cpu().detach() for i in tot_extracts[0]], c = "b")
plt.scatter([i[26].cpu().detach() for i in tot_extracts[1]], [i[52].cpu().detach() for i in tot_extracts[1]], c = "r")
plt.savefig("g12.png")
