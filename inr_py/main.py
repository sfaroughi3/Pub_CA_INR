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


import argparse
import getpass
import imageio
import json
import os
import random
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from siren import Siren
from util import to_coordinates_and_features, to_numpy_features, bpp, model_size_in_bits, plot_GT_Vs_Reconstructed, mean
from training import Trainer


data_folder = Path("/Users/kshadi/Desktop/siren-metascape/data")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
    parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=50000)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
    parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
    parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
    parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
    parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
    parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
    parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
    parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)

    args = parser.parse_args()

    # Set up torch and cuda
    dtype = torch.float32
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

    # Set random seeds
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    if args.full_dataset:
        min_id, max_id = 1, 24  # Kodak dataset runs from kodim01.png to kodim24.png
    else:
        min_id, max_id = args.image_id, args.image_id

    # Dictionary to register mean values (both full precision and half precision)
    results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

    # Create directory to store experiments
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Fit images
    for i in range(min_id, max_id + 1):
    #    print(f'Image {i}')

        dim_out = 3
        # Load from file
        fileDir = data_folder / 'implicit_first_try.xlsx'
        df, arrGT = to_numpy_features(fileDir, 100, 600, 200, 700, dim_out)
        mesh = transforms.ToTensor()(arrGT).float().to(device, dtype)
        print(mesh.shape)
        print(mesh)
        img = mesh

        # Setup model
        func_rep = Siren(
            dim_in=2,
            dim_hidden=args.layer_size,
            dim_out=dim_out,
            num_layers=args.num_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=args.w0_initial,
            w0=args.w0
        ).to(device)

        # Set up training
        trainer = Trainer(func_rep, lr=args.learning_rate)
        coordinates, features = to_coordinates_and_features(img)
        coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
        print(coordinates.shape)
        print(features)
        
        # Calculate model size. Divide by 8000 to go from bits to kB
        model_size = model_size_in_bits(func_rep) / 8000.
        print(f'Model size: {model_size:.1f}kB')
        fp_bpp = bpp(model=func_rep, image=img)
        print(f'Full precision bpp: {fp_bpp:.2f}')

        # Train model in full precision
        trainer.train(coordinates, features, num_iters=args.num_iters)
        print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

        # Log full precision results
        results['fp_bpp'].append(fp_bpp)
        results['fp_psnr'].append(trainer.best_vals['psnr'])

        # Save best model
        torch.save(trainer.best_model, args.logdir + f'/best_model_{i}.pt')

        # Update current model to be best model
        func_rep.load_state_dict(trainer.best_model)

        # Save full precision image reconstruction
        with torch.no_grad():
    #        img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], dim_out).permute(2, 0, 1)
    #        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/fp_reconstruction_{i}.png')
            
            arrRec = func_rep(coordinates).cpu().reshape(img.shape[1],  img.shape[2], dim_out).permute(2, 0, 1).numpy().swapaxes(2,0)
            plot = plot_GT_Vs_Reconstructed(df,arrGT,arrRec)

    print('Full results:')
    print(results)
    with open(args.logdir + f'/results.json', 'w') as f:
        json.dump(results, f)

    # Compute and save aggregated results
    results_mean = {key: mean(results[key]) for key in results}
    with open(args.logdir + f'/results_mean.json', 'w') as f:
        json.dump(results_mean, f)

    print('Aggregate results:')
    print(f'Full precision, bpp: {results_mean["fp_bpp"]:.2f}, psnr: {results_mean["fp_psnr"]:.2f}')
    print(f'Half precision, bpp: {results_mean["hp_bpp"]:.2f}, psnr: {results_mean["hp_psnr"]:.2f}')
