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
import os
import numpy as np
import sys
import pickle as pk

newp = os.path.join(os.path.dirname(os.getcwd()), "/src") 
sys.path.insert(0, os.path.dirname(os.getcwd()))

# Get the current filename
current_filename = os.path.basename(__file__)

# Get the parent directory of the current filename
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_directory = os.path.dirname(current_directory)

# Add the parent's parent directory to the system path
sys.path.append(parent_directory)

from inr_py.siren import Siren
from inr_py.training import Trainer
from inr_py.loader import Loader


# Set up torch and cuda
dtype = torch.float32
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

if __name__ == "__main__":
    ncontext = 1
    #data_root = "E:\Jonathan\Climate_new\src\scripts\data"
    data_root = "data"
    data = np.load(os.path.join(data_root,'data6by2.npy'))
    L = Loader(data, device, ncontext)
    performance = {}
    
    # Jonathan was here, check std
    model_sizes_2 = []
    model_dims_2 = []
    
    # Create the directory if it doesn't exist
    os.makedirs('./models_data6by2', exist_ok=True)

    for numLayers in range(10,13):#for numLayers in range(5, 15, 1):
        for numNodes in range(40, 51,5):#for numNodes in range(5, 71, 5):
            model_sizes_2.append(numLayers * numNodes)
            model_dims_2.append((numLayers, numNodes))

    #Sorted by largest to smallest model
    sorted_model_dims_2 = [dim for _, dim in sorted(zip(model_sizes_2, model_dims_2), key=lambda x: x[0], reverse=True)]
    # sorted_model_dims_2 = sorted_model_dims_2[24:]

    for (numLayer, numNodes) in sorted_model_dims_2:
        print("Num Layers: ", numLayer, "Num Neurons: ", numNodes)
        representation = Siren(
                dim_in= len(data.shape) -1 + ncontext,
                dim_hidden=numNodes,
                dim_out=data.shape[-1] - ncontext,
                num_layers=numLayer,
                final_activation=torch.nn.Identity(),
            )#.to(device)
        representation = torch.nn.DataParallel(representation)
        representation.to(device)

        T = Trainer(L, representation) #func_rep)
        T.train(30000)

        performance[(numLayer,numNodes)] = T.logs
        tc = torch.save(T.best_model,f'./models_data6by2/model-{numLayer}-{numNodes}')
        with open('./models_data6by2/Need_Loss.pk','wb') as f:
            pk.dump(performance, f)



