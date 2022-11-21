https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 8/30/20 8:49 PM


class MaxEntClassifier:
  def __init__(self, labels, num_features):
    self.labels = labels
    self.num_labels = len(labels)
    self.num_features = num_features

  ################################### TRAIN ####################################
  def train(self,
            instances,
            dev_instances=None,
            learning_rate=0.001,
            batch_size=64,
            num_iter=50):
    """TODO: train MaxEnt model with mini-batch stochastic gradient descent.
        Feel free to add more hyperparmaeters as necessary."""
    raise NotImplementedError()

  ################################## COMPUTE ###################################
  def compute_log_posterior(self):
    """TODO: Calculate log posterior"""
    raise NotImplementedError()

  def compute_posterior(self):
    """TODO: Calculate posterior. Please read the instructions carefully for
            numerically stable implementation trick"""
    raise NotImplementedError()

  def compute_gradient(self):
    """TODO: Calculate gradient for a single data instance"""
    raise NotImplementedError()

  def compute_neg_log_likelihood(self):
    """TODO: Calculate negative log likelihood"""
    raise NotImplementedError()

  ################################## PREDICT ###################################
  def classify(self, instance):
    """TODO: predict the most likely label for the given data instance"""
    raise NotImplementedError()
