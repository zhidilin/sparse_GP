import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from GPmodules.GPs import SVGP_new
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
num_inducing_points = 8

inducing_points_ini = torch.randn(num_inducing_points, train_x.size(1)) * 0.5
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SVGP_new(inducing_points=inducing_points_ini)
if not model.variational_strategy.variational_params_initialized.item():
    prior_dist = model.variational_strategy.prior_distribution
    model.variational_strategy._variational_distribution.initialize_variational_distribution(prior_dist)
    model.variational_strategy.variational_params_initialized.fill_(1)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

num_epochs = 1000
optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": 0.01},
                              {'params': likelihood.parameters(), "lr": 0.01},
                              ])
model.train()
# Our loss object.
epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    for x_batch, y_batch in train_loader:
        # 手动清缓存
        model.variational_strategy._clear_cache()
        optimizer.zero_grad()
        loss = model.titsias25_loss(x_batch, y_batch, likelihood)
        epochs_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

model.eval()
plot_snelson(model, likelihood, train_x, train_y, test_x, inducing_points_ini, color='green')
