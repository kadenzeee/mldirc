#!/usr/bin/env python3

import sys
import os
import ROOT  # type: ignore ::: need to be in environment with ROOT installed and sourced
import numpy as np
import time

SHUFFLE = True

program_start = time.time()

ROOT.gInterpreter.ProcessLine('#include "../../../prttools/PrtTools.h"')

try:
    ROOT.gSystem.Load("../../../prtdirc/build/libPrt.dylib")
except FileNotFoundError():
    ROOT.gSystem.Load("../../../prtdirc/build/libPrt.so")

infile = "../../../Data/Raw/500K22TO90DEG.root"
if(len(sys.argv) > 1):
    infile = sys.argv[1] 

t = ROOT.PrtTools(infile)
entries = t.entries()
nchan = t.npix() * t.npmt()


TIMES  = np.zeros((entries, nchan))
ANGLES = np.zeros((entries, 7))
LABELS = np.zeros(entries)

while t.next() and t.i() < entries:

    if not bool(t.event().getHits()): # Skips empty events
        continue

    i = t.i()

    times = [photon.getLeadTime() + ROOT.gRandom.Gaus(0, 0.2) for photon in t.event().getHits()]
    chs   = [int(photon.getChannel()) for photon in t.event().getHits()]
    theta = (t.event().getTof() + ROOT.gRandom.Gaus(0, 3E-03)) 
    phi   = (t.event().getTofPhi() + ROOT.gRandom.Gaus(0, 3E-03))
    
    print(t.event().getTofP())

    mu    = np.mean(times)
    std   = np.std(times)
    t0    = np.min(times)
    t1    = np.max(times)

    chind = np.zeros(nchan)
    chind[chs] += 1

    TIMES[i]  = chind
    ANGLES[i] = [mu, std, t0, t1, np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    LABELS[i] = t.pid()/2 - 1


ANGLES.T[0] = (ANGLES.T[0] - np.mean(ANGLES.T[0]))/np.std(ANGLES.T[0])
ANGLES.T[1] = (ANGLES.T[1] - np.mean(ANGLES.T[1]))/np.std(ANGLES.T[1])
ANGLES.T[2] = (ANGLES.T[2] - np.mean(ANGLES.T[2]))/np.std(ANGLES.T[2])
ANGLES.T[3] = (ANGLES.T[3] - np.mean(ANGLES.T[3]))/np.std(ANGLES.T[3])


print('[INFO] Shuffling...')

if SHUFFLE:
    shuffle = np.random.permutation(entries)

    TIMES  = TIMES[shuffle]
    ANGLES = ANGLES[shuffle]
    LABELS = LABELS[shuffle]

print('[INFO] Saving...')

outfile = infile.replace(".root", "broken_angles")
outfile = os.path.basename(outfile)
np.savez_compressed(outfile, TIMES=TIMES, ANGLES=ANGLES, LABELS=LABELS)

program_end = time.time()
print(f"Done in {program_end-program_start} seconds")

