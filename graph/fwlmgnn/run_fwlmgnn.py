#!/usr/bin/env python

'''
Imports a trained GNN model and evaluates it on a given test dataset, printing the accuracy.
'''


import torch
from torch_geometric.loader import DataLoader
from panda_fwlmgnn import PandaGNN

def evaluate(model, dataset, batch_size=1024):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            
            preds = out.argmax(dim=1)
            
            labels = batch.y.view(-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    accuracy = correct / total
    
    print(f"Accuracy: {accuracy:.4f}")
    
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    
    return accuracy, all_preds, all_labels

def plot_confusion_matrix(preds, labels):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    mpl.useTex = True
    mpl.rc('text', usetex = True, )
    mpl.rc('font', family = 'serif', size = 14)
    
    cm = confusion_matrix(y_true=labels, y_pred=preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0.5, 1.5], [r'$\pi$+', r'$K$+'])
    plt.yticks([0.5, 1.5], [r'$\pi$+', r'$K$+'])
    plt.title('Confusion Matrix')
    plt.show()

def plot_sep_theta(preds, labels, dataset):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from accsep import acc_to_sep
    
    mpl.useTex = True
    mpl.rc('text', usetex = True, )
    mpl.rc('font', family = 'serif', size = 14)
    
    mom_vector = [g.global_features[5:8] for g in dataset]

    thetas = [torch.acos(m[2] / torch.norm(m)).item() * 180 / torch.pi + 22 for m in mom_vector if torch.norm(m) > 0.5]
    theta_bins = torch.arange(22.5, 142.5, 5)
    accs = []
    
    for i in range(len(theta_bins)-1):
        bin_preds = [p for p, t in zip(preds, thetas) if theta_bins[i] <= t < theta_bins[i+1]]
        bin_labels = [l for l, t in zip(labels, thetas) if theta_bins[i] <= t < theta_bins[i+1]]
        
        if len(bin_labels) == 0:
            accs.append(0)
        else:
            accs.append(sum(p == l for p, l in zip(bin_preds, bin_labels)) / len(bin_labels))
    
    # Convert accuracies to separation powers
    sep_powers = [acc_to_sep(acc) for acc in accs]

    import pandas as pd 
    
    # compare to dense nets and time imaging results from perf_vs_ti.ipynb
    with open('data/perf_vs_ti.csv', 'r') as f:
        df = pd.read_csv(f)
    ti_x = df['theta_bin_center']
    ti_y = df['ti_sep']
    ti_yerr = df['ti_err']
    film_acc_x = df['theta_bin_center'][:len(df['film_sep'])]
    film_acc_y = df['film_sep'][:len(df['film_sep'])]
    film_acc_yerr = df['film_err'][:len(df['film_sep'])]

    plt.plot(theta_bins[:-1] + 2.5, sep_powers, marker='o', label='GNN')
    plt.plot(film_acc_x, film_acc_y, marker='o', label='FwLM NN', color='orange')
    plt.errorbar(ti_x, ti_y, yerr=ti_yerr, fmt='o', label='Time Imaging', color='green')
    plt.xlabel(r'Polar Angle, $\theta$ (°)')
    plt.ylabel(r'Separation Power, $\sigma$')
    
    plt.grid(which='minor', linewidth=0.6, ls='--')
    plt.minorticks_on()
    
    plt.legend()
    plt.ylim(0, None)
    plt.xlim(20, 145)
    plt.title(r'Model Accuracy vs. $\theta$')
    plt.grid()
    plt.show()

def plot_sep_mom(preds, labels, dataset):
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from accsep import acc_to_sep
    
    mpl.useTex = True
    mpl.rc('text', usetex = True, )
    mpl.rc('font', family = 'serif', size = 14)

    moms = [g.global_features[4] for g in dataset]
    mom_bins = torch.arange(0.5, 3.6, 0.2)
    accs = []
    
    for i in range(len(mom_bins)-1):
        bin_preds = [p for p, m in zip(preds, moms) if mom_bins[i] <= m < mom_bins[i+1]]
        bin_labels = [l for l, m in zip(labels, moms) if mom_bins[i] <= m < mom_bins[i+1]]
        
        if len(bin_labels) == 0:
            accs.append(0)
        else:
            accs.append(sum(p == l for p, l in zip(bin_preds, bin_labels)) / len(bin_labels))
    
    # Convert accuracies to separation powers
    sep_powers = [acc_to_sep(acc) for acc in accs]

    plt.plot(mom_bins[:-1] + 0.25, sep_powers, marker='o')
    plt.xlabel(r'Momentum (GeV/c)')
    plt.ylabel(r'Separation Power, $\sigma$')
    
    plt.grid(which='minor', linewidth=0.6, ls='--')
    plt.minorticks_on()
    
    plt.title(r'Model Accuracy vs. Momentum')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    
    # ----- CLI ----- #
    
    import argparse
    
    parser = argparse.ArgumentParser(prog='run_gnn', description='Evaluates GNN on given test data.')
    
    parser.add_argument('-im', '--model_input', type=str, required=True, help='Path to input model weights.')
    parser.add_argument('-id', '--data_input', type=str, required=True, help='Path to input .pkl file to run tests on.')
    parser.add_argument('--plot-cm', action='store_true', help='Plot confusion matrix after evaluation.')
    parser.add_argument('--plot-sep-theta', action='store_true', help='Plot model accuracy vs. theta after evaluation.')
    parser.add_argument('--plot-sep-mom', action='store_true', help='Plot model accuracy vs. momentum after evaluation.')

    args = parser.parse_args()
    
    # ----- Load ----- #
    
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
    
    model = PandaGNN(node_dim=3, edge_dim=3, global_dim=8, hidden_dim=128, n_classes=2)
    model = torch.compile(model)
    model.load_state_dict(torch.load(args.model_input, map_location=torch.device('cpu')))
    
    accuracy, all_preds, all_labels = evaluate(model, dataset)
    
    if args.plot_cm: plot_confusion_matrix(all_preds, all_labels)
    if args.plot_sep_theta: plot_sep_theta(all_preds, all_labels, dataset)
    if args.plot_sep_mom: plot_sep_mom(all_preds, all_labels, dataset)