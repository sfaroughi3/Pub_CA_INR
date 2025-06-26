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

from inr_py.util import get_clamped_psnr
from inr_py.loader import Loader


class Infer():
    def __init__(self, representation: nn.Module, gt: np.ndarray, device: str, ncontext = 0):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.loader = Loader(gt, device, ncontext)
        self.ncontext = ncontext
        self.shape = self.loader.data.shape

    def infer(self, denormalize = True):
        """return numpy array with the shape of ground truth image.
        """
        predicted = self.representation(self.loader.coordinates) # this causes the issue 
        data = predicted


        if denormalize:
            for i,scale in enumerate(self.loader.scales[self.ncontext:]):
                mn, mx = scale
                minn = data[:, i].min()
                maxx = data[:, i].max()
                data[:, i] = (data[:,i] - minn)/(maxx-minn)
                data[:,i] = (mx-mn)*data[:,i] + mn
        return data.reshape(self.shape)
