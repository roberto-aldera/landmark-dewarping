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

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].grid()
    ax[0].plot(scores.detach().numpy()[0], '.', label="Scores")
    ax[0].plot(torch.sort(scores)[0].detach().numpy()[0], '.', label="Sorted scores by magnitude")
    ax[0].plot(sorted_indices[0], scores.detach().numpy()[0], '.', label="Scores in order of sorted thetas")
    # ax[0].plot(sorted_thetas[0], '.', label="Sorted thetas")
    ax[0].legend()

    ax[1].grid()
    # ax[1].plot(sorted_thetas.detach().numpy()[0], '.')
    scores = scores.detach().numpy()[0]
    thetas_np = thetas.detach().numpy()[0]
    threshold = 2.75
    candidates = np.where(scores > threshold)
    inliers = scores[candidates]

    # ax[1].plot(scores, '.', label="Scores")
    ax[1].plot(thetas_np[candidates], '.', label="Thetas")
    # ax[1].plot(thetas_np, '.', label="All thetas")
    # ax[1].plot(np.sort(thetas_np), '.', label="Sorted thetas")
    # ax[1].set_ylim(-0.01, 0.01)
    # ax[1].set_xlim(0, 5)
    ax[1].legend()

    fig.suptitle("Debugging scores and their corresponding thetas")
    fig.savefig("%s%s" % (settings.RESULTS_DIR, "debugging.pdf"))
    plt.close()
    print("Saved figure to:", "%s%s" % (settings.RESULTS_DIR, "debugging.pdf"))
    pdb.set_trace()
