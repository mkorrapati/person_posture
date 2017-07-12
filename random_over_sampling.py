#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:25:27 2017

@author: muralikorrapati

====================
Random over-sampling
====================
An illustration of the random over-sampling method.

# Authors: Christos Aridas
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from imblearn.over_sampling import RandomOverSampler

def random_over_sample(X, y):
    # Instanciate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=2)
    # Fit and transform x to visualise inside a 2D feature space
    X_vis = pca.fit_transform(X)
    
    # Apply the random over-sampling
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_sample(X, y)
    X_res_vis = pca.transform(X_resampled)

    return X_resampled, y_resampled, X_vis, X_res_vis

def visualize_over_sample_data(X, y, X_resampled, y_resampled, X_vis, X_res_vis):
    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)
    
    c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0",
                     alpha=0.5)
    c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1",
                     alpha=0.5)
    ax1.set_title('Original set')
    
    ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
                label="Class #0", alpha=.5)
    ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
                label="Class #1", alpha=.5)
    ax2.set_title('Random over-sampling')
    
    # make nice plotting
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-6, 8])
        ax.set_ylim([-6, 6])
    
    plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center',
                  ncol=2, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.show()