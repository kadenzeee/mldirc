#!/usr/bin/env python

'''
Imports a set of trained GNN models and evaluates them on a given test dataset, printing the accuracy.
'''


import torch
import numpy as np
from torch_geometric.loader import DataLoader
from panda_fwlmgnn import PandaGNN, PandaGNNLayer
import torch.nn.functional as F

def evaluate(model, dataset, batch_size=1024):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    
    correct = 0
    total = 0
    
    total_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            
            labels = batch.y.view(-1).long()
            preds = out.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.append(preds)
            all_labels.append(labels)
            
            loss = F.cross_entropy(out, labels, reduction='sum')
            total_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = total_loss / total
    
    print(f"Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    
    return accuracy, avg_loss, all_preds, all_labels


import re

def extract_epoch(filename):
    match = re.search(r"epoch-(\d+)", filename)
    return int(match.group(1)) if match else -1



if __name__ == "__main__":
    
    # ----- CLI ----- #
    
    import argparse, os, re
    
    parser = argparse.ArgumentParser(prog='run_gnn', description='Evaluates GNN on given test data.')
    
    parser.add_argument('-im', '--model_input', type=str, required=True, help='Path to input model weights; directory should contain model_epoch_*.pt files.')
    parser.add_argument('-id', '--data_input', type=str, required=True, help='Path to input .pkl file to run tests on.')
    parser.add_argument('--save-npz', action='store_true', help='Save predictions and labels to .npz file for further analysis.')

    args = parser.parse_args()
    
    import pickle
    from torch_geometric.data import Data

    with open(args.data_input, "rb") as f:
        data = pickle.load(f)

    nevents = len(data['labels'])

    dataset = []

    for i in range(nevents):

        x           = torch.tensor(data['nodes'][i], dtype=torch.float)
        edge_index  = torch.tensor(data['edges'][i].T, dtype=torch.long)
        edge_attr   = torch.tensor(data['edge_features'][i], dtype=torch.float)

        y   = torch.tensor([data['labels'][i]], dtype=torch.long)
        g   = torch.tensor(data['globals'][i], dtype=torch.float) 

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph.global_features = g
        
        dataset.append(graph)
    
    model_files = [
    f for f in os.listdir(args.model_input) if f.endswith(".pt")
    ]

    model_files = sorted(model_files, key=extract_epoch)
    
    results = []
    epochs = []
    
    print(f"[WARN] Loading and evaluating model files requires the execution of .pt files, which may contain arbitrary code. Ensure trust in the source of model files.")
    
    for model_file in model_files:
        print(f"Evaluating model: {model_file}")
        model_path = os.path.join(args.model_input, model_file)
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        config = checkpoint['model_config']
        
        model = PandaGNN(**config)

        model.load_state_dict(checkpoint['model_state_dict'])
        
        accuracy, loss, all_preds, all_labels = evaluate(model, dataset)
        
        train_loss = checkpoint["train_loss"]
        epoch = checkpoint["epoch"]
        results.append((model_file, accuracy, loss, train_loss))
        epochs.append(epoch)
    
    filenames, accuracies, val_losses, train_losses = zip(*results)

    accuracies = list(accuracies)
    val_losses = list(val_losses)
    train_losses = list(train_losses)
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    mpl.useTex = True
    mpl.rc('text', usetex = True, )
    mpl.rc('font', family = 'serif', size = 14)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(epochs, accuracies, marker='o')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Validation Accuracy')
    axs[0].set_title('Validation Accuracy vs. Epoch')
    axs[0].grid()
    
    axs[1].plot(epochs, val_losses, marker='o', label='Validation Loss')
    axs[1].plot(epochs, train_losses, marker='o', label='Training Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_yscale('log')
    axs[1].set_title('Loss vs. Epoch')
    axs[1].legend()
    axs[1].grid()
    

    plt.tight_layout()
    plt.savefig(f"{args.model_input}/validation_results.png")
    
    if args.save_npz: np.savez(f"{args.model_input}/validation_results.npz", epochs=epochs, accuracies=accuracies, val_losses=val_losses, train_losses=train_losses)