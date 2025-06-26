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


import pdb
import torch
import os
import numpy as np
import sys
import pickle as pk
import pandas as pd
import gc
sys.path.insert(0, os.path.dirname(os.getcwd()))

from inr_py.siren import Siren
from inr_py.infer import Infer


# Set up torch and cuda
dtype = torch.float32
device = 'cuda'


if __name__ == "__main__":
    data_root = "data"
    df1 = pd.DataFrame()
    ncontext = 0
    
    data = np.load(os.path.join(data_root, 'data6by1.npy'))
    for layer in reversed(range(5,15)):   #for layer in reversed(range(5, 15)):   
        for unit in reversed(range(40,71,5)):  #for unit in reversed(range(5, 75, 5)):  
            print(layer, unit)
            model = f"./models_data6by1/model-{layer}-{unit}"   #"./models/model-10-20"
            num_layers = int(model.split('/')[-1].split('-')[1])
            dim_hidden = int(model.split('/')[-1].split('-')[2])
            _representation = torch.load(model)  # .eval()
            representation = Siren(
                  dim_in=len(data.shape) - 1 + ncontext,
                  dim_hidden=dim_hidden,
                  dim_out=data.shape[-1] - ncontext,
                  num_layers=num_layers,
                  final_activation=torch.nn.Identity(),
            )#.to(device)
            representation = torch.nn.DataParallel(representation)
            representation.to(device)
            
            representation.load_state_dict(_representation)
            
            with torch.no_grad():
                ig = Infer(representation, data, device, ncontext)
                predicted = ig.infer()
                Predicted = predicted.detach().cpu().numpy()
                #print(Predicted.shape)
                Predicted = Predicted.reshape(720, 1440,6,1)
                np.save(f'Infer_data6by1_{layer}_{unit}.npy'.format(layer, unit), Predicted)
