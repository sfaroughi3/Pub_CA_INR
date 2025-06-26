"""Context-Aware Implicit Neural Representations to Compress Earth Systems Model Data

DEVELOPED AT:
                    Department of Chemical Engineering
                    University of Utah, Salt Lake City,
                    Utah 84112, USA
                    DIRECTOR: Prof.  Salah A Faroughi

DEVELOPED BY:
                    Energy and Intelligence Lab

MIT License

Copyright (c) 2024 Energy and Intelligence Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import tqdm
from collections import OrderedDict
from torch import nn
import numpy as np
import copy

from inr_py.util import get_clamped_psnr
from inr_py.loader import Loader
from inr_py.My_Schedulers import My_StepLR


class Trainer():
    def __init__(self, loader: Loader,representation: nn.Module, lr=1e-3, print_freq=1, step_size=1000,gamma=0.5 ):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        #self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=1e-3) # JONATHAN WAS HERE
        #self.lr_2 = 1e-4

        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())
        self.loader = loader
        self.scheduler = My_StepLR(lr=lr, step_size=step_size, gamma=gamma)

    def train(self, num_iters):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        phase_2_epoch_start = num_iters // 2
        copied_model_state_dict = None
        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Update model
                self.optimizer.zero_grad()
                predicted = self.representation(self.loader.coordinates)
                loss = self.loss_func(predicted, self.loader.data)
                loss.backward()
                self.optimizer.step()

                self.scheduler.step()
                if self.scheduler.update_lr():
                    self.representation.load_state_dict(copied_model_state_dict)
                    self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=self.scheduler.get_lr())
                    #print(f"Updated Lr: {self.scheduler.get_lr()}")

                # Calculate psnr
                psnr = get_clamped_psnr(predicted, self.loader.data)

                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)

                    copied_model_state_dict = copy.deepcopy(self.representation.state_dict()) # Jonathan Code
