#!/usr/bin/env python3

import numpy as np
import os
import glob

# ------------------------------------------------------------
# Configure shapes (must match conversion script)
# ------------------------------------------------------------

nbins = 10        # must match training config
npmt  = 16        # adjust if needed
npix  = 64        # adjust if needed

nchan = npmt * npix
hist_size = nbins * npmt
angle_size = 7

# ------------------------------------------------------------
# Find chunk files
# ------------------------------------------------------------

hist_files  = sorted(glob.glob("*_HISTS.dat"))
time_files  = sorted(glob.glob("*_TIMES.dat"))
angle_files = sorted(glob.glob("*_ANGLES.dat"))
label_files = sorted(glob.glob("*_LABELS.dat"))

assert len(hist_files) > 0, "No chunk files found"
assert len(hist_files) == len(time_files) == len(angle_files) == len(label_files)

print(f"[INFO] Found {len(hist_files)} chunks")

# ------------------------------------------------------------
# Determine total entries
# ------------------------------------------------------------

chunk_sizes = []
for f in hist_files:
    size = os.path.getsize(f)
    entries = size // hist_size   # int8 = 1 byte
    chunk_sizes.append(entries)

total_entries = sum(chunk_sizes)

print(f"[INFO] Total entries: {total_entries}")

# ------------------------------------------------------------
# Create final memmaps
# ------------------------------------------------------------

HISTS  = np.memmap("HISTS_full.dat",  dtype=np.int8,    mode='w+', shape=(total_entries, hist_size))
TIMES  = np.memmap("TIMES_full.dat",  dtype=np.float16, mode='w+', shape=(total_entries, nchan))
ANGLES = np.memmap("ANGLES_full.dat", dtype=np.float16, mode='w+', shape=(total_entries, angle_size))
LABELS = np.memmap("LABELS_full.dat", dtype=np.int8,    mode='w+', shape=(total_entries,))

# ------------------------------------------------------------
# Merge sequentially
# ------------------------------------------------------------

offset = 0

for i in range(len(hist_files)):

    print(f"[INFO] Merging chunk {i+1}/{len(hist_files)}")

    chunk_n = chunk_sizes[i]

    h = np.memmap(hist_files[i],  dtype=np.int8,    mode='r', shape=(chunk_n, hist_size))
    t = np.memmap(time_files[i],  dtype=np.float16, mode='r', shape=(chunk_n, nchan))
    a = np.memmap(angle_files[i], dtype=np.float16, mode='r', shape=(chunk_n, angle_size))
    l = np.memmap(label_files[i], dtype=np.int8,    mode='r', shape=(chunk_n,))

    HISTS[offset:offset+chunk_n]  = h
    TIMES[offset:offset+chunk_n]  = t
    ANGLES[offset:offset+chunk_n] = a
    LABELS[offset:offset+chunk_n] = l

    offset += chunk_n

# ------------------------------------------------------------
# Flush
# ------------------------------------------------------------

HISTS.flush()
TIMES.flush()
ANGLES.flush()
LABELS.flush()

print("[INFO] Merge complete")
