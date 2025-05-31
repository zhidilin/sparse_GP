from matplotlib import pyplot as plt
import numpy as np
import torch

def plot_snelson(model, likelihood, train_x, train_y, test_x, inducing_points_ini=None):
    """
    Plot the Snelson dataset with GP predictions and confidence intervals.

    Parameters:
    - model: The trained GP model.
    - likelihood: The likelihood function for the GP.
    - train_x: Training input data.
    - train_y: Training output data.
    - test_x: Test input data for predictions.
    - inducing_points_ini: Initial inducing points for the GP.
    """

    # Make predictions
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        preds = likelihood(model(test_x))

    # Prediction mean and confidence intervals
    preds_mean = preds.mean
    preds_std = preds.stddev
    preds_lower = preds_mean - 1.96 * preds_std
    preds_upper = preds_mean + 1.96 * preds_std

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(test_x.cpu().numpy(),
             preds_mean.cpu().numpy(),
             label='Predicted Mean', color='purple')
    plt.fill_between(test_x.cpu().squeeze().numpy(),
                     preds_lower.cpu().numpy(),
                     preds_upper.cpu().numpy(),
                     color='purple', alpha=0.15, label='95% Confidence Interval')
    plt.scatter(train_x.cpu().numpy(), train_y.cpu().numpy(),
                label='Training Data', color='k', s=20)

    # Inducing points
    if inducing_points_ini is not None:
        inducing_points = model.variational_strategy.inducing_points.detach().cpu().numpy()
        inducing_points_ini = inducing_points_ini.detach().cpu().numpy()  # you must have saved this earlier
        # Plot inducing points
        y_max = preds_upper.max().item()
        y_min = preds_lower.min().item()
        y_top = y_max + 0.05 * (y_max - y_min)
        y_bot = y_min - 0.05 * (y_max - y_min)

        plt.plot(inducing_points_ini,
                 np.full_like(inducing_points_ini, y_top),
                 '+', color='black')
        plt.plot(inducing_points, np.full_like(inducing_points, y_bot),
                 '+', color='black')

    # Limit x-range to test_x domain
    plt.xlim(test_x.min().item(), test_x.max().item())

    # Hide axes
    plt.xticks([])
    plt.yticks([])
    # for spine in plt.gca().spines.values():
    #     spine.set_visible(False)
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    plt.tight_layout()
    plt.show()