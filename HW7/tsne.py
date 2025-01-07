#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab
import matplotlib.pyplot as plt
from PIL import Image
import os

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, mode=0, threshold=1e-5):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)
    
    img = []
    prev_cost = None

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        if not mode:        
            # t-SNE
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            # symmetric SNE
            num = np.exp(-1. * np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        if not mode:
            # t-SNE
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        else:
            # symmetric SNE
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)


        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 50 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            save_embedding_with_gif(Y, labels, mode, perplexity, iter + 1, img)
            if prev_cost is not None and abs(prev_cost - C) < threshold:
                print(f"Converged at iteration {iter + 1} with error {C:.6f}")
                break
            prev_cost = C

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    
    plot_similarity(P, Q, mode, perplexity)
    return Y, img

def save_embedding_with_gif(Y, labels, mode, perplexity, iter_num, image_files):
    """
    Saves the current embedding as a frame and appends it for GIF creation.
    """
    plt.figure(figsize=(8, 8))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(Y[idx, 0], Y[idx, 1], label=f'Class {label}', alpha=0.7, s=10)
    # plt.legend()
    plt.title(f"Iteration {iter_num}: {'t-SNE' if mode == 0 else 'Symmetric SNE'}_Perplexity={perplexity}")
    plt.axis('off')
    filename = f"{'t-SNE' if mode == 0 else 'Symmetric SNE'}_Perplexity{perplexity}_{iter_num:04d}.png"
    plt.savefig(filename)
    plt.close()
    image_files.append(filename)

def create_gif_and_save_final(image_files, output_gif, final_output, duration=100):
    """
    Creates a GIF from captured frames and saves the final embedding as a static image.
    """
    # Create GIF from all frames
    frames = [Image.open(img) for img in image_files]
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"GIF saved as {output_gif}")

    # Save the last frame as the final embedding
    if image_files:
        os.rename(image_files[-1], final_output)
        print(f"Final embedding saved as {final_output}")

    # Clean up intermediate frame images
    for img in image_files[:-1]:
        os.remove(img)

def plot_similarity(P, Q, mode=0, perplexity=10):
    """
    Plots the distribution of pairwise similarities for high-dimensional and low-dimensional spaces.
    
    Parameters:
    - P: Pairwise similarities in high-dimensional space.
    - Q: Pairwise similarities in low-dimensional space.
    - mode: 0 for t-SNE, 1 for symmetric SNE.
    - perplexity: Perplexity value used for labeling and saving the plot.
    """
    plt.subplot(2, 1, 1)
    plt.title(f'{"Symmetric SNE" if mode else "t-SNE"} high-dim (Perplexity={perplexity})')
    plt.hist(P.flatten(), bins=40, log=True)

    plt.subplot(2, 1, 2)
    plt.title(f'{"Symmetric SNE" if mode else "t-SNE"} low-dim (Perplexity={perplexity})')
    plt.hist(Q.flatten(), bins=40, log=True)

    plt.tight_layout()
    filename = f'{"symmetric_SNE" if mode else "t-SNE"}_Perplexity{perplexity}.jpg'
    plt.savefig(filename)
    plt.close()
    print(f"Saved similarity plot: {filename}")


if __name__ == "__main__":
    """
    Main function:
        Run multiple combinations of perplexity and mode for t-SNE or Symmetric SNE.
    """

    print('===== Reading data =====')
    x = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")

    perplexitys = [10, 20, 30, 40, 50]  # 不同的 perplexity 值
    modes = [0, 1]  # 0: t-SNE, 1: Symmetric SNE

    for pp in perplexitys:
        for m in modes:
            print(f'=== Start {"t-SNE" if not m else "Symmetric SNE"} with perplexity={pp} ===')
            try:
                y, img = tsne(x, no_dims=2, initial_dims=50, perplexity=pp, mode=m)
                
                output_gif = f'{"t-SNE" if not m else "symmetric SNE"}_{pp}.gif'
                final_output = f"{"t-SNE" if not m else "symmetric SNE"}_{pp}.png"
                create_gif_and_save_final(img, output_gif=output_gif, final_output=final_output)

                print(f"Saved GIF: {output_gif} and final image: {final_output}")
            except ValueError as e:
                print(f"Error during SNE with mode={m}, perplexity={pp}: {e}")

    print("=== All tasks completed ===")
