"""
    Author: geraltofrivia

    Script to use to model to do basic stuff like choosing b/w given set of paths etc.
"""
import os
import numpy as np
from keras.models import load_model

from network import custom_loss as loss_fn

DEFAULT_MODEL_DIR = 'data/training/pairwise/model_28'


class ModelInterpreter:

    def __init__(self, _model_dir = DEFAULT_MODEL_DIR):
        """
            Use this object for anything that has to do with the trained model.
            @TODO: Describe most major functions here.

        """

        # Find and load the model from disk.
        self.model = load_model(os.path.join(_model_dir, 'model.h5'), custom_objects={'custom_loss': loss_fn})
        self._parse_model_inputs()

    def _parse_model_inputs(self):
        """
            Function that would parse the model's config to parse input dimensions.

        :return: None
        """
        config = self.model.get_config()

        input_shapes = []   # End goal: fill this.

        # Sift through it and get input shapes
        for layer in config['layers']:

            # Forget all those which aren't an input layer.
            if not layer['class_name'] == 'InputLayer':
                continue

            # For the rest, get batch_input_shape
            input_shapes.append(layer['config']['batch_input_shape'][1:])

        self.embedding_dim = input_shapes[0][1]
        self.path_len = input_shapes[1][0]
        self.question_len = input_shapes[0][0]

    def rank(self, _v_q, _v_ps, _return_indices=False, _k=0):
        """
            Function to evaluate a bunch of paths and return a ranked list

        :param _v_q: vector of dimension (n, 300)
        :param _v_ps: list of vectors, each of dimensions (m, 300)
        :param _k: int: if more than 0, returns cropped results
        :param _return_indices: Boolean, deciding whether to return paths or

        :return: indices, or path vectors
        """
        # Pad paths
        padded_paths = np.zeros((len(_v_ps), self.path_len, self.embedding_dim))
        for i in range(len(_v_ps)):
            _v_p = _v_ps[i]
            padded_paths[i, :_v_p.shape[0], :_v_p.shape[1]] = _v_p

        # Pad question
        padded_ques = np.zeros((len(_v_ps), self.question_len, self.embedding_dim))
        padded_ques[:, :_v_q.shape[0], :_v_q.shape[1]] = np.repeat(
            a=_v_q[np.newaxis, :, :],
            repeats=len(_v_ps),
            axis=0)

        # Create a dummy set of paths for the sake of model arg: input3
        dummy_paths = np.zeros((len(_v_ps), self.path_len, self.embedding_dim))

        # Pass to model.
        similarities = self.model.predict([padded_ques, padded_paths, dummy_paths])[0]

        # Reshape from an array of n arrays of 1 element [[i],[j],[k]] -> [i,j,k]
        similarities = np.transpose(similarities)

        # Reshape it to a 1D array
        similarities = similarities.reshape(similarities.shape[1])

        # Rank
        rank = np.argsort(similarities)[::-1]

        if 0 < _k <= rank.shape[0]:
            # If one needs top-k results or not
            return rank[:_k] if _return_indices else _v_ps[rank[:_k]]
        else:
            return rank if _return_indices else _v_ps[rank]






if __name__ == "__main__":

    model_interpreter = ModelInterpreter()
    vq = np.random.rand(15,300)
    vps = [ np.random.rand(9, 300) for x in range(6)]
    model_interpreter.rank(vq, vps)