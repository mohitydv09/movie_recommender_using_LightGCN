# Movie Recommendation System using LightGCN

This repository contains my implementation of a movie recommendation system using Light Graph Convolutional Networks (GCNs). The system is built using PyTorch Geometric and it utilizes the MovieLens dataset for training and evaluation.

The original paper by Xiangnan He et al., which proposed LightGCN, can be found [here](https://dl.acm.org/doi/abs/10.1145/3397271.3401063).

I followed the blog post [here](https://medium.com/stanford-cs224w/movie-recommender-systems-with-pyg-37da71f405a4) for implementation.

## Overview

The recommendation system employs a Graph Convolutional Network architecture to learn user and movie embeddings from the user-movie interaction graph. It predicts user ratings for movies, facilitating personalized movie recommendations.

## Methodology

LightGCN model was used to learn user and movie embeddings using the MovieLens dataset. Ratings >= 4/5 were considered as positive interactions. Bayesian Personalized Ranking (BPR) loss metric was used to guide the model to learn better embeddings. For evaluation, Recall@K metric was used to measure the proportion of top-K recommendations that the user has already enjoyed.

The model was trained for 50 epochs with a batch size of 256, using the Adam optimizer with a learning rate of 1e-3 and an exponential learning rate scheduler with a decay rate of 0.9.

## Result
After training the LightGCN model, it achieved a recall@20 of 0.12.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgements
This project utilizes the MovieLens dataset provided by the GroupLens Research lab.