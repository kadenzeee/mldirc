#!/usr/bin/env python3
#filepath: mldirc/dense/film/convert_film_fisher_yates.py

import sys
import os
import subprocess
import ROOT  # type: ignore ::: need to be in environment with ROOT installed and sourced
import numpy as np
import time

program_start = time.time()

ROOT.gInterpreter.ProcessLine('#include "../../../prttools/PrtTools.h"')
try:
    ROOT.gSystem.Load("../../../prtdirc/build/libPrt.dylib")
except FileNotFoundError():
    ROOT.gSystem.Load("../../../prtdirc/build/libPrt.so")

os.makedirs("tmp", exist_ok=True)

infile = "2M22TO90.root"
if(len(sys.argv) > 1):
    infile = sys.argv[1] 

t = ROOT.PrtTools(infile)
entries = t.entries()
nchan = t.npix() * t.npmt()
splt = 100000                   # tmp chunksize


TIMES  = np.zeros((entries, nchan))
ANGLES = np.zeros((entries, 7))
LABELS = np.zeros(entries)

s = 0

while t.next() and t.i() < entries:

    if not bool(t.event().getHits()): # Skips empty events
        continue

    i = t.i()

    times = [photon.getLeadTime() + ROOT.gRandom.Gaus(0, 0.2) for photon in t.event().getHits()]
    chs   = [int(photon.getChannel()) for photon in t.event().getHits()]
    theta = (t.event().getTof() + ROOT.gRandom.Gaus(0, 3E-03)) * np.pi/180
    phi   = (t.event().getTofP() + ROOT.gRandom.Gaus(0, 3E-03)) * np.pi/180

    mu    = np.mean(times)
    std   = np.std(times)
    t0    = np.min(times)
    t1    = np.max(times)
    
    times = np.array(times) - mu
    
    chind = np.zeros(nchan)
    chind[chs] += times

    TIMES[i]  = chind
    ANGLES[i] = [mu, std, t0, t1, np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    LABELS[i] = t.pid()/2 - 1
    
    if ANGLES[i][4] == 0 and ANGLES[i][5] == 0 and ANGLES[i][6] == 0:
        print('[WARN] Zero angle detected. Investigate?')
    
    if i % splt == 0 and i != 0:
        start = s * splt
        end = start + splt
        
        ANGLES[start:end].T[0] = (ANGLES[start:end].T[0] - np.mean(ANGLES[start:end].T[0]))/np.std(ANGLES[start:end].T[0])
        ANGLES[start:end].T[1] = (ANGLES[start:end].T[1] - np.mean(ANGLES[start:end].T[1]))/np.std(ANGLES[start:end].T[1])
        ANGLES[start:end].T[2] = (ANGLES[start:end].T[2] - np.mean(ANGLES[start:end].T[2]))/np.std(ANGLES[start:end].T[2])
        ANGLES[start:end].T[3] = (ANGLES[start:end].T[3] - np.mean(ANGLES[start:end].T[3]))/np.std(ANGLES[start:end].T[3])
        
        np.save(f"tmp/times_{s}.npy", TIMES[start:end])
        np.save(f"tmp/angles_{s}.npy", ANGLES[start:end])
        np.save(f"tmp/labels_{s}.npy", LABELS[start:end])
        
        s += 1

start = s * splt
if start < entries:
    end = entries
    
    ANGLES[start:end].T[0] = (ANGLES[start:end].T[0] - np.mean(ANGLES[start:end].T[0]))/np.std(ANGLES[start:end].T[0])
    ANGLES[start:end].T[1] = (ANGLES[start:end].T[1] - np.mean(ANGLES[start:end].T[1]))/np.std(ANGLES[start:end].T[1])
    ANGLES[start:end].T[2] = (ANGLES[start:end].T[2] - np.mean(ANGLES[start:end].T[2]))/np.std(ANGLES[start:end].T[2])
    ANGLES[start:end].T[3] = (ANGLES[start:end].T[3] - np.mean(ANGLES[start:end].T[3]))/np.std(ANGLES[start:end].T[3])
    
    np.save(f"tmp/times_{s}.npy", TIMES[start:end])
    np.save(f"tmp/angles_{s}.npy", ANGLES[start:end])
    np.save(f"tmp/labels_{s}.npy", LABELS[start:end])

# Load all tmp files
all_time_files = sorted([f"tmp/{f}" for f in os.listdir("tmp") if f.startswith("times_")])
all_angle_files = sorted([f"tmp/{f}" for f in os.listdir("tmp") if f.startswith("angles_")])
all_label_files = sorted([f"tmp/{f}" for f in os.listdir("tmp") if f.startswith("labels_")])

# Flattened memmaps
TIMES_MM = np.memmap("TIMES_full.npy", dtype=np.float64,  mode='w+', shape=((entries, nchan)))
ANGLES_MM = np.memmap("ANGLES_full.npy", dtype=np.float64, mode='w+', shape=((entries, 7)))
LABELS_MM = np.memmap("LABELS_full.npy", dtype=np.int64,  mode='w+', shape=(entries,))

print('[INFO] Memmaps created, writing flattened arrays...')

# Merge tmp files and write
offset = 0
for fname in all_time_files:
    chunk = np.load(fname, mmap_mode='r')
    L = len(chunk)
    TIMES_MM[offset:offset+L] = chunk
    offset += L

print('[INFO] Time array complete...')

offset = 0
for fname in all_angle_files:
    chunk = np.load(fname, mmap_mode='r')
    L = len(chunk)
    ANGLES_MM[offset:offset+L] = chunk
    for i in ANGLES_MM[offset:offset+L]:
        if i[4] == 0 and i[5] == 0 and i[6] == 0:
            print('[WARN] Zero angle detected. Investigate?')
    
    offset += L
    
print('[INFO] Angle array complete...')

offset = 0
for fname in all_label_files:
    chunk = np.load(fname, mmap_mode='r')
    L = len(chunk)
    LABELS_MM[offset:offset+L] = chunk
    offset += L

print('[INFO] Label array complete...')

TIMES_MM.flush()
ANGLES_MM.flush()
LABELS_MM.flush()

print('[INFO] Tmp files merged, shuffling array...')
subprocess.run("rm -r tmp", shell=True)

# Fisher-Yates shuffle

rng = np.random.default_rng()

for i in range(entries - 1, 0, -1):
    j = rng.integers(0, i + 1)

    # ---- Swap times ----
    tmp = TIMES_MM[i].copy()
    TIMES_MM[i] = TIMES_MM[j]
    TIMES_MM[j] = tmp

    # ---- Swap angles ----
    tmp = ANGLES_MM[i].copy()
    ANGLES_MM[i] = ANGLES_MM[j]
    ANGLES_MM[j] = tmp
    
    # ---- Swap labels ----
    tmp = LABELS_MM[i].copy()
    LABELS_MM[i] = LABELS_MM[j]
    LABELS_MM[j] = tmp

TIMES_MM.flush()
ANGLES_MM.flush()
LABELS_MM.flush()

print('[INFO] Shuffle complete')

outfile = infile.replace(".root", "timing")
outfile = os.path.basename(outfile)
np.savez_compressed(f"{outfile}.npz", TIMES=TIMES_MM, ANGLES=ANGLES_MM, LABELS=LABELS_MM)

subprocess.run("rm TIMES_full.npy", shell=True)
subprocess.run("rm ANGLES_full.npy", shell=True)
subprocess.run("rm LABELS_full.npy", shell=True)

program_end = time.time()
print(f"Done in {program_end-program_start} seconds")
