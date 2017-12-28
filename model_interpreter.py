"""
    Author: geraltofrivia

    Script to use to model to do basic stuff like choosing b/w given set of paths etc.
"""
import os
import keras
import numpy as np
from keras.models import load_model

from network import custom_loss as loss_fn

DEFAULT_MODEL_DIR = 'data/training/pairwise/model_00'


class ModelInterpreter:

    def __init__(self, _model_dir = DEFAULT_MODEL_DIR):
        """
            Use this object for anything that has to do with the trained model.
            @TODO: Describe most major functions here.

        """

        # Find and load the model from disk.
        self.model = load_model(os.path.join(_model_dir,'model.h5'), custom_objects={'custom_loss': loss_fn})
        self.embedding_dim = 300
        self.max_q_tokens = 4
        self.max_p_tokens = 4

    def rank(self, _v_q, _v_ps, _return_indices=False, _k=0):
        """
            Function to evaluate a bunch of paths and return a ranked list

        :param _v_q: vector of dimension (n, 300)
        :param _v_ps: list of vectors, each of dimensions (m, 300)
        :param _k: int: if more than 0, returns cropped results
        :param _return_indices: Boolean, deciding whether to return paths or

        :return: indices, or path vectors
        """
        # Pad question @TODO
        # Pad paths @TODO

        dummy_path = np.zeroes((self.max_p_tokens, self.embedding_dim))