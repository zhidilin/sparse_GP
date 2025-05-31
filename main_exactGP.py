import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from GPmodules.GPs import ExactGPModel
from utils.setup import reset_seed
from utils.utils_dataset import DataNormalizer, load_Snelson
from utils.SnelsonPlots import plot_snelson
reset_seed(4560)

# load the data
train_x, train_y, test_x = load_Snelson('dataset/Snelson.mat')
# Normalize the data
normalizer_x = DataNormalizer(train_x)
normalizer_y = DataNormalizer(train_y)
train_x = normalizer_x.normalize(train_x)
test_x = normalizer_x.normalize(test_x)
train_y = normalizer_y.normalize(train_y)
if torch.cuda.is_available():
    train_x, train_y, test_x = train_x.cuda(), train_y.cuda(), test_x.cuda()

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

num_epochs = 1000
model.train()
likelihood.train()
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.01)

# Our loss object.
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()
    print()

plot_snelson(model, likelihood, train_x, train_y, test_x, color='orange')
