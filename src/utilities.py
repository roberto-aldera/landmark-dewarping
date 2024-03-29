import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import settings
import torch
import pdb


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for name, parameter in named_parameters:
        if parameter.requires_grad and ("bias" not in name):
            layers.append(name)
            ave_grads.append(parameter.abs().mean().detach().numpy())
            max_grads.append(parameter.abs().max().detach().numpy())
    plt.figure(figsize=(10, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("%s%s%i%s" % (settings.RESULTS_DIR, "gradients/", settings.GRAD_PLOTTING_ITR, "_gradients.pdf"))
    plt.close()


def plot_scores_and_thetas(scores, thetas):
    sorted_thetas, sorted_indices = torch.sort(thetas)
    scores = scores.detach().numpy()[0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0, 0].grid()
    ax[0, 0].plot(scores, '.', label="Scores")
    ax[0, 0].plot(np.sort(scores), '.', label="Sorted scores by magnitude")
    ax[0, 0].legend()

    ax[0, 1].grid()
    thetas_np = thetas.detach().numpy()[0]
    threshold = 0.5
    candidates = np.where(scores > threshold)
    inliers = scores[candidates]
    theta_candidates = np.empty(len(thetas_np))
    theta_candidates[:] = np.nan
    theta_candidates[candidates] = thetas_np[candidates]

    ax[0, 1].plot(thetas_np, '.', label="All thetas")
    ax[0, 1].plot(theta_candidates, '.', label="Candidate thetas")
    # ax[1].set_ylim(-0.01, 0.01)
    # ax[1].set_xlim(0, 5)
    ax[0, 1].legend()

    ax[1, 0].grid()
    ax[1, 0].plot(torch.argsort(sorted_indices[0]), thetas[0], '.', label="Sorted thetas")
    ax[1, 0].plot(torch.argsort(sorted_indices[0]), scores, '.', label="Scores in order of sorted thetas")
    ax[1, 0].legend()

    ax[1, 1].grid()
    ax[1, 1].plot(theta_candidates[candidates], '.', label="Candidate thetas")
    ax[1, 1].legend()

    fig.suptitle("Debugging scores and their corresponding thetas")
    fig.savefig("%s%s" % (settings.RESULTS_DIR, "debugging.pdf"))
    plt.close()
    print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "debugging.pdf"))
    pdb.set_trace()


def plot_thetas_in_batch(thetas):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0, 0].grid()
    ax[0, 0].plot(thetas[0:30], '.')

    ax[0, 1].grid()
    ax[0, 1].plot(np.mean(thetas.detach().numpy(), axis=1), '.', label="Means of each batch")
    ax[0, 1].legend()

    ax[1, 0].grid()
    ax[1, 0].hist(np.abs(np.mean(thetas.detach().numpy(), axis=1)), density=True, bins=30)

    ax[1, 1].grid()
    ax[1, 1].plot(np.abs(np.mean(thetas.detach().numpy(), axis=1)), '.', label="Abs means of each batch")
    ax[1, 1].text(0, 0.1, ("%s%s" % ("Mean = ", str(np.mean(np.abs(np.mean(thetas.detach().numpy(), axis=1)))))))
    ax[1, 1].legend()

    fig.suptitle("Debugging thetas")
    fig.savefig("%s%s" % (settings.RESULTS_DIR, "debugging-thetas.pdf"))
    plt.close()
    print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "debugging-thetas.pdf"))
    pdb.set_trace()


def plot_quantiles_and_thetas(predicted_quantiles, thetas):
    quantile_width = 0.75
    quantiles = torch.tensor([0.5 - (quantile_width / 2), 0.5 + (quantile_width / 2)], dtype=torch.float32)
    real_quantiles = torch.quantile(thetas, quantiles)

    inner_thetas = thetas[((thetas >= real_quantiles[0]) & (thetas <= real_quantiles[1]))]
    predicted_inner_thetas = thetas[((thetas >= predicted_quantiles[0, 0]) & (thetas <= predicted_quantiles[0, 1]))]

    sorted_thetas, sorted_indices = torch.sort(thetas)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].grid()
    ax[0].plot(sorted_thetas[0], '.', label="sorted thetas")
    ax[0].plot(inner_thetas, '.', label="inner thetas")
    ax[0].legend()

    ax[1].grid()
    ax[1].plot(predicted_inner_thetas, '.', label="predicted inner thetas")
    # ax[1].set_ylim(-0.01, 0.01)
    # ax[1].set_xlim(0, 5)
    ax[1].legend()

    fig.suptitle("Debugging scores and their corresponding thetas")
    fig.savefig("%s%s" % (settings.RESULTS_DIR, "debugging-quantiles.pdf"))
    plt.close()
    print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "debugging-quantiles.pdf"))
    pdb.set_trace()


def plot_theta_clusters(predicted_quantiles, thetas):
    quantile_width = 0.75
    quantiles = torch.tensor([0.5 - (quantile_width / 2), 0.5 + (quantile_width / 2)], dtype=torch.float32)
    real_quantiles = torch.quantile(thetas, quantiles)

    inner_thetas = thetas[((thetas >= real_quantiles[0]) & (thetas <= real_quantiles[1]))]
    predicted_inner_thetas = thetas[((thetas >= predicted_quantiles[0, 0]) & (thetas <= predicted_quantiles[0, 1]))]

    sorted_thetas, sorted_indices = torch.sort(thetas)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=8, random_state=0, tol=1e-6).fit(thetas[0].reshape(-1, 1))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].grid()
    ax[0].plot(sorted_thetas[0], '.', label="sorted thetas")
    ax[0].plot(inner_thetas, '.', label="inner thetas")
    ax[0].legend()

    ax[1].grid()
    ax[1].plot(thetas[0][kmeans.labels_ == 0], '.', label="c0")
    ax[1].plot(thetas[0][kmeans.labels_ == 1], '.', label="c1")
    ax[1].plot(thetas[0][kmeans.labels_ == 2], '.', label="c2")
    ax[1].plot(thetas[0][kmeans.labels_ == 3], '.', label="c3")
    ax[1].plot(thetas[0][kmeans.labels_ == 4], '.', label="c4")
    ax[1].plot(thetas[0][kmeans.labels_ == 5], '.', label="c5")
    ax[1].plot(thetas[0][kmeans.labels_ == 6], '.', label="c6")
    ax[1].plot(thetas[0][kmeans.labels_ == 7], '.', label="c7")
    # ax[1].set_ylim(-0.01, 0.01)
    # ax[1].set_xlim(0, 5)
    ax[1].legend()

    fig.suptitle("Debugging scores and their corresponding thetas")
    fig.savefig("%s%s" % (settings.RESULTS_DIR, "debugging-clustering.pdf"))
    plt.close()
    print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "debugging-clustering.pdf"))
    pdb.set_trace()
