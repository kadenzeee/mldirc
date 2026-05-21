#!/usr/bin/env python3
#filepath: mldirc/graph/convert_nf.py

'''
Converts prtdirc simulation data into a format suitable for training a NF network.
'''

import argparse
import platform
import time
import ROOT #type:ignore
import numpy as np

program_start = time.time()

# ----- CLI ----- # 

parser = argparse.ArgumentParser(prog='convert_nf', description='Converts prtdirc simulation data into a format suitable for training a NF network.')

parser.add_argument('-i', '--input', type=str, required=True, help='Path to input ROOT file.')
parser.add_argument('-o', '--output', type=str, required=False, help='Path to output .npz file. If not provided, defaults to nf_data.npz in the current directory.')
parser.add_argument('-max-photons', '--max-photons', type=int, default=64, help='Maximum number of photons to consider per event. Default: 128')

args = parser.parse_args()

ROOT.gInterpreter.ProcessLine('#include "../../../prttools/PrtTools.h"')

libbase = "../../../prtdirc/build/libPrt"
libpath = libbase + (".dylib" if platform.system()=="Darwin" else ".so")
ROOT.gSystem.Load(libpath)

# ----- Load ----- #

f = ROOT.PrtTools(args.input)
entries = f.entries()

# ----- Run ----- #

max_photons = args.max_photons

events_x    = np.zeros((entries, max_photons, 3), dtype=np.float32) # photon features: x, y, lead time
events_c    = np.zeros((entries, 3), dtype=np.float32) # event features: momentum, TOF, TOF_pi
events_y    = np.zeros(entries, dtype=np.int32) # event labels
events_mask = np.zeros((entries, max_photons), dtype=np.bool_) # mask to indicate valid photons

j = 0
while f.next() and j < entries:

    if not f.event().getHits():
        continue
    
    hits = f.event().getHits()
    
    # ------ PID ----- #
    
    pid = f.event().getPid() - 2 # minus 2 because prtdirc simulation labels Pi+ : 2 and Kaon+ : 3
    
    # ------ Features ----- #
    
    p           = f.event().getMomentum()
    mag_p       = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    
    if mag_p == 0:
        continue
    
    i = 0
    for hit in hits:
        
        if i >= max_photons:
            break
        
        x = [
            hit.getPosition()[0], hit.getPosition()[1], hit.getLeadTime()
        ]
        
        c = [
            mag_p, f.event().getTof(), f.event().getTofPi()
        ]
        
        events_x[j, i] = x
        events_c[j] = c
        events_y[j] = pid
        events_mask[j, i] = True
        
        i += 1
    
    j += 1


events_x = events_x[:j]
events_c = events_c[:j]
events_y = events_y[:j]

# ----- Standardise ----- #

events_x = np.where((events_x == 0), np.nan, events_x)

events_x[:,:,0] = (events_x[:,:,0] - np.nanmean(events_x[:,:,0])) / np.nanstd(events_x[:,:,0])    
events_x[:,:,1] = (events_x[:,:,1] - np.nanmean(events_x[:,:,1])) / np.nanstd(events_x[:,:,1])
events_x[:,:,2] = (events_x[:,:,2] - np.nanmean(events_x[:,:,2])) / np.nanstd(events_x[:,:,2])

events_x = np.where(np.isnan(events_x), 0, events_x)

events_c = np.where((events_c == 0), np.nan, events_c)

events_c[:,0] = (events_c[:,0] - np.mean(events_c[:,0])) / np.std(events_c[:,0])
events_c[:,1] = (events_c[:,1] - np.mean(events_c[:,1])) / np.std(events_c[:,1]) 
events_c[:,2] = (events_c[:,2] - np.mean(events_c[:,2])) / np.std(events_c[:,2]) 

events_c = np.where(np.isnan(events_c), 0, events_c)

# ----- Save ----- #
output_path = args.output if args.output else "nf_data.npz"

np.savez(
    output_path,
    x=events_x,
    c=events_c,
    y=events_y,
    mask=events_mask
)