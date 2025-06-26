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

import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch




class Loader(Dataset):
    """
        Load dataframe for one ticker at a time
        ncontexts are the number of dimensions that provide context (must be the first dimensions)
    """
    def __init__(self, data: np.ndarray, device, ncontext = 0):
        self.data = torch.tensor(np.float32(data)).to(device)
        dims = data.shape[-1]
        scales = [None]*dims
        for i in range(dims):
            mn = self.data[...,i].min()
            mx = self.data[...,i].max()
            self.data[...,i] = (self.data[...,i]-mn)/(mx-mn)
            scales[i] = (mn,mx)
        self.scales = scales
        coordinates = torch.ones(data.shape[0:-1]).nonzero(as_tuple=False).to(device).to(torch.float32)
        mxcols = coordinates.max(dim=0,keepdim=True).values
        mxcols[mxcols == 0] = 1
        coordinates = coordinates/mxcols
        self.coordinates = 2*(coordinates)-1
        
        if(ncontext > 0):
            contexts = self.data[...,0:ncontext].reshape((coordinates.size(0),ncontext))
            contexts = 2 * contexts - 1
            self.coordinates = torch.cat((self.coordinates, contexts),-1)
            
        self.data = self.data[...,ncontext:].reshape((coordinates.size(0),dims-ncontext))
