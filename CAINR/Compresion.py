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

import numpy as np
import torch
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio
import os
import sys

sys.path.insert(0, os.path.dirname(os.getcwd()))

from inr_py.siren import Siren
from inr_py.infer import Infer

def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * 32 for t in tensors)
               for tensors in (model.parameters(), model.buffers()))

def calculate_psnr(original, reconstructed):
    """Calculate PSNR between original and reconstructed data"""
    
    if torch.is_tensor(original):
        
        original = original.detach().cpu().numpy()
    if torch.is_tensor(reconstructed):
        
        reconstructed = reconstructed.detach().cpu().numpy()
    
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    
    #data_range = max(original.max() - original.min(), 
    #                reconstructed.max() - reconstructed.min())
    
    return peak_signal_noise_ratio(original, reconstructed, 
                                   data_range=max(reconstructed.max(), original.max())) #, data_range=data_range

# Initialize lists to store results
compression_gains = []
psnrs = []
layers_used = []
units_used = []

# Load data
data_root = "./data"
data = np.load(os.path.join(data_root, 'data6by2.npy'))
data_PSNR = np.load(os.path.join(data_root, 'data6by1.npy'))
#data_PSNR = data_PSNR.detach().cpu().numpy()
# Set up torch and cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ncontext = 1

for layer in reversed(range(5,15)):   
    for unit in reversed(range(5,71,5)):  
        #if layer == 14 and unit == 70:
        #    print("Skipping layer=14, unit=70")
        #    continue
            
        print(f"Processing layer={layer}, unit={unit}")
        model_path = f"models_data6by2/model-{layer}-{unit}"
        
        try:
            # Load model
            _representation = torch.load(model_path, map_location=device)
            
            # Initialize model
            representation = Siren(
                dim_in=len(data.shape) - 1 + ncontext,
                dim_hidden=unit,
                dim_out=data.shape[-1] - ncontext,
                num_layers=layer,
                final_activation=torch.nn.Identity(),
            )
            representation = torch.nn.DataParallel(representation)
            representation.to(device)
            representation.load_state_dict(_representation)
            representation.eval()
            
            # Calculate compression gain
            predicted_model_size = model_size_in_bits(representation) / 8000
            #original_size = (data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3] * 32) / 8000
            original_size = (data.shape[0] * data.shape[1] * data.shape[2] * 1 * 64) / 8000
            compression_gain = original_size / predicted_model_size
            print(compression_gain)
            
            # Reconstruct data
            with torch.no_grad():
                ig = Infer(representation, data, device, ncontext)
                predicted = ig.infer()
                Predicted = predicted.detach().cpu().numpy()
                Predicted = predicted.reshape(720, 1440, 6, 1)
                
                
                psnr_value = calculate_psnr(
                    data_PSNR, 
                    Predicted
                )
            print(psnr_value)
            
            # Store results
            compression_gains.append(compression_gain)
            psnrs.append(psnr_value)
            layers_used.append(layer)
            units_used.append(unit)
            
            print(f"Layer: {layer}, Units: {unit}, Compression: {compression_gain:.2f}, PSNR: {psnr_value:.2f}")
            
        except Exception as e:
            print(f"Error processing {model_path}: {str(e)}")
            continue

# Save numpy arrays
#Name = "INR"
Name = "CAINR_TO"
#Name = "CAINR_MT"
#Name = "2CAINR"
np.save("compression_"+Name+".npy", np.array(compression_gains))
np.save("psnr_"+Name+".npy", np.array(psnrs))

# Create DataFrame with results
results_df = pd.DataFrame({
    'compression_gain': compression_gains,
    'psnr': psnrs,
    'num_layers': layers_used,
    'dim_hidden': units_used
})

results_df.to_csv(Name + '.csv', index=False)
