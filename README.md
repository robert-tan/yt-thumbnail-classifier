# Classifying YouTube Videos from Thumbnails

YouTube videos are often classified and tagged with certain video categories in order to be used in YouTube's recommender systems and listed when a user searches videos related to a certain category. The goal of this project is to train a deep learning model to classify the type of the YouTube video based only on its thumbnail. This project can help largely reduce users' cognitive load by auto-classifying the YouTube video. Besides, it can also serve as an input or supplement to other classifiers that classify videos based on other features.

## Dataset

We are using a subset of the YT-8M dataset with per-video features. We have processed the dataset to store thumbnail images as a `(n_w, n_h, 3)`-dimensional array and labeled each thumbnail with one of 25 video categories. The total size of the (processed) dataset is 150GB with around 300,000 examples.

## Model
