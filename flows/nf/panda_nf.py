# filepath: mldirc/nf/model.py

import os

import torch
import torch.nn as nn

from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.normalization import ActNorm

from nflows.nn.nets import ResidualNet


# ============================================================
# FiLM Conditioning
# ============================================================

class FiLM(nn.Module):

    def __init__(self, cond_dim, hidden_dim):
        super().__init__()

        self.gamma = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.beta = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, c):

        # x : (B, Nphotons, hidden_dim)
        # c : (B, cond_dim)

        gamma = self.gamma(c).unsqueeze(1)
        beta  = self.beta(c).unsqueeze(1)

        return gamma * x + beta


# ============================================================
# Transformer Encoder
# ============================================================

class PhotonTransformer(nn.Module):

    def __init__(
        self,
        photon_dim=3,
        cond_dim=3,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ):

        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(photon_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.film = FiLM(cond_dim, d_model)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def masked_mean_pool(self, x, mask):

        # x    : (B, N, D)
        # mask : (B, N)

        mask = mask.unsqueeze(-1)

        x = x * mask

        summed = x.sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1e-6)

        return summed / counts

    def forward(self, photons, cond, mask):

        # photons : (B, Nphotons, 3)
        # cond    : (B, 3)
        # mask    : (B, Nphotons)

        x = self.embedding(photons)

        # PyTorch transformer expects:
        # True = ignore token

        padding_mask = (mask == 0)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.film(x, cond)

        x = self.output_proj(x)

        event_embedding = self.masked_mean_pool(x, mask)

        return event_embedding


# ============================================================
# Conditional Flow
# ============================================================

class ConditionalFlowModel(nn.Module):

    def __init__(
        self,
        photon_dim=3,
        cond_dim=3,
        max_photons=64,
        latent_dim=128,
        num_flow_steps=6
    ):

        super().__init__()

        self.encoder = PhotonTransformer(
            photon_dim=photon_dim,
            cond_dim=cond_dim,
            d_model=latent_dim
        )

        transforms = []

        for _ in range(num_flow_steps):

            transforms.append(
                ReversePermutation(features=latent_dim)
            )

            transforms.append(
                ActNorm(features=latent_dim)
            )

            transforms.append(
                AffineCouplingTransform(
                    mask=self._create_alternating_mask(latent_dim),
                    transform_net_create_fn=lambda in_features, out_features:
                        ResidualNet(
                            in_features=in_features,
                            out_features=out_features,
                            hidden_features=256,
                            num_blocks=2,
                            activation=torch.relu
                        )
                )
            )

        transform = CompositeTransform(transforms)

        base_distribution = StandardNormal([latent_dim])

        self.flow = Flow(transform, base_distribution)

    def _create_alternating_mask(self, dim):

        mask = torch.arange(dim) % 2

        return mask.float()

    def encode(self, photons, cond, mask):

        return self.encoder(
            photons,
            cond,
            mask
        )

    def log_prob(self, photons, cond, mask):

        z = self.encode(
            photons,
            cond,
            mask
        )

        return self.flow.log_prob(z)

    def sample(self, n_samples):

        return self.flow.sample(n_samples)

    def forward(self, photons, cond, mask):

        return self.log_prob(
            photons,
            cond,
            mask
        )
        

import numpy as np
import torch

from torch.utils.data import Dataset


class DIRCDataset(Dataset):

    def __init__(self, folder):

        self.files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npz")
        ])
        
        x_list = []
        c_list = []
        mask_list = []
        y_list = []
        
        data_bar = tqdm.tqdm(enumerate(self.files), total=len(self.files), desc='Building dataset')
        
        for file_id, file in data_bar:

            data = np.load(file)

            x_list.append(data["x"])
            c_list.append(data["c"])
            mask_list.append(data["mask"])
            y_list.append(data["y"])

        self.x = torch.from_numpy(
            np.concatenate(x_list, axis=0)
        ).float()

        self.c = torch.from_numpy(
            np.concatenate(c_list, axis=0)
        ).float()

        self.mask = torch.from_numpy(
            np.concatenate(mask_list, axis=0)
        ).float()

        self.y = torch.from_numpy(
            np.concatenate(y_list, axis=0)
        ).long()

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        return {
            "photons": self.x[idx],
            "cond": self.c[idx],
            "mask": self.mask[idx],
            "pid": self.y[idx]
        }

# ============================================================

if __name__ == "__main__":
    
    import argparse, tqdm, time, subprocess
    
    parser = argparse.ArgumentParser(prog='convert_nf', description='Converts PrtTools ROOT files to NumPy arrays for NF training.')
    
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input .npz file\\s.')
    parser.add_argument('-o', '--output', type=str, required=False, default='tmp', help='Path to output model file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--batch', action='store_true', help='Whether to use batch training (default: False).')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ConditionalFlowModel().to(device)
    
    dataset = DIRCDataset(args.input)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    if args.batch:
        raise NotImplementedError()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )
    
    for epoch in range(args.epochs):
        
        pbar = tqdm.tqdm(loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            
            t1 = time.time()
            
            photons = batch["photons"].to(device)
            cond    = batch["cond"].to(device)
            mask    = batch["mask"].to(device)
            
            optimizer.zero_grad()
            
            log_prob = model.log_prob(
                photons,
                cond,
                mask
            )
            
            loss = -log_prob.mean()
            
            loss.backward()
            
            optimizer.step()
            
            t2 = time.time()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'compute': f"{(t2 - t1):.2f}s"
            })
        
        checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "config":
                vars(args),

            "loss":
                loss.item()
        }
        
        subprocess.run("mkdir -p {}".format(args.output), shell=True, check=True)
        
        torch.save(
            checkpoint,
            os.path.join(
                args.output, f"epoch_{epoch:04d}.pt"
            )       
        )