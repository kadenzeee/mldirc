#!/usr/bin/env python3
#filepath: mldirc/dense/film/convert_film_fisher_yates.py

import os
import subprocess
import ROOT  # type: ignore ::: need to be in environment with ROOT installed and sourced
import numpy as np
import time
import argparse
import sys
import shutil
import signal
import json
import platform
try:
    import psutil
except Exception:
    psutil = None

program_start = time.time()

parser = argparse.ArgumentParser(
    prog='convert_dualfilm',
    description='Convert ROOT data to numpy arrays for dualfilm training')

parser.add_argument('-i', '--input', help='Input ROOT file path', default='500K22TO140.root')
parser.add_argument('-n', '--nbins', type=int, help='Number of histogram bins', default=10)
parser.add_argument('-tsmear', '--tsmear', type=float, help='Standard deviation of Gaussian noise added to photon arrival time data, in the same units as the input times (e.g. ns)', default=0.1)

args = parser.parse_args()

ROOT.gInterpreter.ProcessLine('#include "../../../prttools/PrtTools.h"')


libbase = "../../../prtdirc/build/libPrt"

if platform.system() == "Darwin":
    libpath = libbase + ".dylib"
else:
    libpath = libbase + ".so"

if not os.path.exists(libpath):
    raise FileNotFoundError(f"Library not found: {libpath}")

ret = ROOT.gSystem.Load(libpath)

if ret != 0:
    raise RuntimeError(f"ROOT failed to load {libpath}")

os.makedirs("tmp", exist_ok=True)


t = ROOT.PrtTools(args.input)
entries = t.entries()
nchan = t.npix() * t.npmt()
splt = 100000           # tmp chunksize

infile = args.input
nbins = args.nbins
tedges = np.linspace(-30, 30, nbins+1)
mcpedges = np.arange(t.npmt()+1)*t.npix()

# --- logging, preflight checks and helpers ---------------------------------
LOGFILE = "convert_dualfilm.log"

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f"[{ts}] {msg}\n"
    print(entry, end='')
    try:
        with open(LOGFILE, 'a') as f:
            f.write(entry)
    except Exception:
        pass

def get_free_disk_bytes(path='.'):
    try:
        du = shutil.disk_usage(path)
        return du.free
    except Exception:
        return None

def get_free_memory_bytes():
    try:
        if psutil is not None:
            return psutil.virtual_memory().available
        if platform.system() == 'Darwin':
            out = subprocess.check_output(['vm_stat']).decode('utf-8')
            pagesize = 4096
            free = 0
            inactive = 0
            for line in out.splitlines():
                if 'Pages free' in line:
                    free = int(line.split(':')[1].strip().replace('.', ''))
                if 'Pages inactive' in line:
                    inactive = int(line.split(':')[1].strip().replace('.', ''))
            return (free + inactive) * pagesize
    except Exception:
        return None

def estimate_bytes_for_arrays(entries, nchan, nbins, npmt):
    bytes_times = int(entries) * int(nchan) * 4
    bytes_hists = int(entries) * (int(nbins) * int(npmt)) * 4
    bytes_angles = int(entries) * 7 * 4
    bytes_labels = int(entries) * 1
    return bytes_times + bytes_hists + bytes_angles + bytes_labels

def handle_termination(signum, frame):
    log(f"Received signal {signum}; terminating early")
    try:
        with open('convert_status.json', 'w') as f:
            json.dump({'status': 'terminated', 'signal': int(signum)}, f)
    except Exception:
        pass
    sys.exit(1)

for s in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(s, handle_termination)
    except Exception:
        pass

# Preflight: estimate memory & disk and warn/abort if insufficient
estimated_needed = estimate_bytes_for_arrays(entries, nchan, nbins, t.npmt())
free_mem = get_free_memory_bytes()
free_disk = get_free_disk_bytes('.')
log(f"Estimated RAM needed for arrays: {estimated_needed/1e9:.3f} GB")
if free_mem is not None:
    log(f"Free memory: {free_mem/1e9:.3f} GB")
    if free_mem < estimated_needed * 1.2:
        log("WARNING: Available memory is less than estimated need; this may cause OOM kills")

if free_disk is not None:
    log(f"Free disk: {free_disk/1e9:.3f} GB")
    # memmaps and tmp files could use a lot; require at least 2x of estimated
    if free_disk < estimated_needed * 2:
        log("WARNING: Low disk space for memmaps and tmp files; aborting to avoid partial writes")
        raise SystemExit("Insufficient disk space")

# Allocate main arrays (in memory)
TIMES  = np.zeros((entries, nchan))
HISTS  = np.zeros((entries, nbins*t.npmt()))
ANGLES = np.zeros((entries, 7))
LABELS = np.zeros(entries)

s = 0

tstart = time.time()

while t.next() and t.i() < entries:

    if not bool(t.event().getHits()): # Skips empty events
        continue

    i = t.i()

    times = [photon.getLeadTime() + ROOT.gRandom.Gaus(0, args.tsmear) for photon in t.event().getHits()]
    chs   = [int(photon.getChannel()) for photon in t.event().getHits()]
    theta = (t.event().getTof() + ROOT.gRandom.Gaus(0, 3E-03)) * np.pi/180
    phi   = (t.event().getTofP() + ROOT.gRandom.Gaus(0, 3E-03)) * np.pi/180


    mu    = np.mean(times)
    std   = np.std(times)
    t0    = np.min(times)
    t1    = np.max(times)
    
    times = (np.array(times) - mu)
    
    chind = np.zeros(nchan)
    chind[chs] += times

    HISTS[i] = np.histogram2d(times, chs, bins=[tedges, mcpedges])[0].flatten()
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
        
        np.save(f"tmp/hists_{s}.npy", HISTS[start:end])
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
    
    np.save(f"tmp/hists_{s}.npy", HISTS[start:end])
    np.save(f"tmp/times_{s}.npy", TIMES[start:end])
    np.save(f"tmp/angles_{s}.npy", ANGLES[start:end])
    np.save(f"tmp/labels_{s}.npy", LABELS[start:end])


# Feature-wise standardisation for angle branch

ANGLES = (ANGLES - np.mean(ANGLES, axis=0)) / np.std(ANGLES, axis=0)

tend = time.time()
print(f"[INFO] Event processing done in {tend-tstart} seconds")

# Load all tmp files
all_hist_files = sorted([f"tmp/{f}" for f in os.listdir("tmp") if f.startswith("hists_")])
all_time_files = sorted([f"tmp/{f}" for f in os.listdir("tmp") if f.startswith("times_")])
all_angle_files = sorted([f"tmp/{f}" for f in os.listdir("tmp") if f.startswith("angles_")])
all_label_files = sorted([f"tmp/{f}" for f in os.listdir("tmp") if f.startswith("labels_")])

# Flattened memmaps.
# 8-bit signed are much much cheaper on disk, and doesn't matter in terms of speed because we are not performing any computations on these arrays anymore.
# Similarly for float16, which is sufficient precision for NN.

HISTS_MM = np.memmap("HISTS_full.npy", dtype=np.int8, mode='w+', shape=((entries, nbins*t.npmt())))
TIMES_MM = np.memmap("TIMES_full.npy", dtype=np.float16,  mode='w+', shape=((entries, nchan)))
ANGLES_MM = np.memmap("ANGLES_full.npy", dtype=np.float16, mode='w+', shape=((entries, 7)))
LABELS_MM = np.memmap("LABELS_full.npy", dtype=np.int8,  mode='w+', shape=(entries,))

print('[INFO] Memmaps created, writing flattened arrays...')

# Merge tmp files and write
offset = 0
for fname in all_hist_files:
    chunk = np.load(fname, mmap_mode='r')
    L = len(chunk)
    HISTS_MM[offset:offset+L] = chunk
    offset += L
    subprocess.run(f"rm {fname}", shell=True)       # cleans up behind itself

print('[INFO] Histogram array complete...')

offset = 0
for fname in all_time_files:
    chunk = np.load(fname, mmap_mode='r')
    L = len(chunk)
    TIMES_MM[offset:offset+L] = chunk
    offset += L
    subprocess.run(f"rm {fname}", shell=True)

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
    subprocess.run(f"rm {fname}", shell=True)

print('[INFO] Angle array complete...')

offset = 0
for fname in all_label_files:
    chunk = np.load(fname, mmap_mode='r')
    L = len(chunk)
    LABELS_MM[offset:offset+L] = chunk
    offset += L
    subprocess.run(f"rm {fname}", shell=True)

print('[INFO] Label array complete...')

HISTS_MM.flush()
TIMES_MM.flush()
ANGLES_MM.flush()
LABELS_MM.flush()

print('[INFO] Tmp files merged, shuffling array...')
subprocess.run("rm -r tmp", shell=True)

# Fisher-Yates shuffle

rng = np.random.default_rng()

for i in range(entries - 1, 0, -1):
    j = rng.integers(0, i + 1)

    # ---- Swap histograms ----
    tmp = HISTS_MM[i].copy()
    HISTS_MM[i] = HISTS_MM[j]
    HISTS_MM[j] = tmp

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

HISTS_MM.flush()
TIMES_MM.flush()
ANGLES_MM.flush()
LABELS_MM.flush()

print('[INFO] Shuffle complete')

outfile = infile.replace(".root", f"dualfilm_{args.tsmear}ns")
outfile = os.path.basename(outfile)
np.savez_compressed(f"{outfile}.npz", TIMES=TIMES_MM, HISTS=HISTS_MM, ANGLES=ANGLES_MM, LABELS=LABELS_MM, logs=args.__dict__)

subprocess.run("rm TIMES_full.npy", shell=True)
subprocess.run("rm HISTS_full.npy", shell=True)
subprocess.run("rm ANGLES_full.npy", shell=True)
subprocess.run("rm LABELS_full.npy", shell=True)

program_end = time.time()
print(f"Done in {program_end-program_start} seconds")
