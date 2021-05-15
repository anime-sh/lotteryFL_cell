import torch
import torch.nn as nn
import torch.optim as optim
from torch import Module
import numpy as np
import os


class Client():
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.model = model()

        # TODO

    def update(
        self,
        globalModel,
        *args,
        **kwargs
    ) -> None:
        """
            Interface to Server
        """
        pass

    def learn(
        self,
        *args,
        **kwargs
    ):
        """
            Train NN
        """
        pass

    def prune(
        self,
        *args,
        **kwargs
    ):
        """
            Prune self.model
        """

        pass

    def eval(
        self,
        *args,
        **kwargs
    ):
        """
            Eval self.model
        """
        pass

    def save(
        self,
        *args,
        **kwargs
    ):
        """
            Save model,meta-info,states
        """
        pass

    def upload(
        self,
        *args,
        **kwargs
    ) -> torch.Module:
        """
            Upload self.model
        """
        pass
