#%%
# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import linregress

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
msize = 20
matplotlib.rc('font', **font)
figsize = (16, 12)

filename = r'efr32bg22IQ_2020-8-17_19-50-30.pickle'
with open(os.path.join(os.getcwd(), '../data', filename), 'rb') as F:
    (I_raw, Q_raw) = pickle.load(F)

n = len(I_raw)
assert n == len(Q_raw)

magnitudes = np.array([i**2 + q**2 for i, q in zip(I_raw, Q_raw)])
start_indices = []
stop_indices = []
thresh = np.mean(magnitudes)
seeking_start = True
for i in range(n):
    if seeking_start and (magnitudes[i] > thresh):
        seeking_start = False
        start_indices.append(i)
    elif not(seeking_start) and (magnitudes[i] < thresh):
        seeking_start = True
        stop_indices.append(i)
if start_indices[-1] > stop_indices[-1]:
    del start_indices[-1]
if stop_indices[0] < start_indices[0]:
    del stop_indices[0]
assert len(start_indices) == len(stop_indices)
max_interval = np.max(np.array([stop-start for start, stop in zip(start_indices, stop_indices)]))
i = 0
while i < len(start_indices):
    if stop_indices[i]-start_indices[i] < .8*max_interval:
        del start_indices[i]
        del stop_indices[i]
    else:
        i += 1

plt.figure(figsize=figsize)
plt.plot(range(n), magnitudes, color='blue', label='Magnitude')
plt.plot(range(n), np.mean(magnitudes)*np.ones(n), color='orange', label='Threshold')
plt.vlines(start_indices, 0, 1.1*np.max(magnitudes[start_indices[0]:stop_indices[-1]]), color='green', label='Rising edges')
plt.vlines(stop_indices, 0, 1.1*np.max(magnitudes[start_indices[0]:stop_indices[-1]]), color='red', label='Falling edges')
plt.xlim([0, n-1])
plt.ylim([0, 1.1*np.max(magnitudes[start_indices[0]:stop_indices[-1]])])
plt.title('Squared magnitude of signal, vs. measurement number')
plt.xlabel('Measurement number')
plt.ylabel('Squared magnitude')
plt.legend()

mean_jump = np.mean((np.mean([s2-s1 for s1, s2 in zip(start_indices[:-1], start_indices[1:])]),
                     np.mean([s2-s1 for s1, s2 in zip(stop_indices[:-1], stop_indices[1:])])))
mean_duration = np.mean([stop-start for start, stop in zip(start_indices, stop_indices)])
midpoints = [start_indices[0]+int(.5*mean_duration)+i*int(mean_jump) for i in range(len(start_indices))]
intervals = []
for midpoint in midpoints:
    intervals.append((midpoint-int(.25*mean_duration), midpoint+int(.25*mean_duration)))
I = []
Q = []
for interval in intervals:
    I.append(I_raw[interval[0]:interval[1]])
    Q.append(Q_raw[interval[0]:interval[1]])

plt.figure(figsize=figsize)
colors = cm.get_cmap('tab20', len(intervals))
colors = [colors(i/(len(intervals))) for i in range(len(intervals))]
lim = 1.1*np.sqrt(np.max(magnitudes[start_indices[0]:stop_indices[-1]]))
for interval, color in zip(intervals, colors):
    plt.axvspan(interval[0], interval[1], alpha=.5, color=color)
plt.plot(range(n), I_raw, '.', markersize=msize, color='red', label='I')
plt.plot(range(n), Q_raw, '.', markersize=msize, color='blue', label='Q')
plt.xlim([np.max((0, start_indices[0]-50)), np.min((n-1, stop_indices[-1]+50))])
plt.ylim([-lim, lim])
plt.title('I/Q values vs. measurement number')
plt.xlabel('Measurement number')
plt.ylabel('I/Q value')
plt.legend()

plt.figure(figsize=figsize)
for i, q, color in zip(I, Q, colors):
    plt.plot(i, q, '.', markersize=msize, color=color)
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Q vs. I')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-lim, lim])
plt.ylim([-lim, lim])

phases = [np.unwrap(np.arctan2(q, i))/np.pi for i, q in zip(I, Q)]
lsrls = [(linregress(range(len(phase)), phase)) for phase in phases]
plt.figure(figsize=figsize)
for phase, (m, b, _, _, _), color in zip(phases, lsrls, colors):
    plt.plot(range(len(phase)), phase, '.', markersize=msize, color=color)
    plt.plot(range(len(phase)), m*np.arange(len(phase))+b, '--', color=color)
plt.xlabel('Measurement number')
plt.ylabel('Phase (radians/pi)')
plt.title('Unwrapped phases of IQ measurements, vs. measurement number')

plt.figure(figsize=figsize)
intercepts = [lsrl[1] for lsrl in lsrls]
plt.plot(range(len(intercepts)), intercepts)
plt.xlabel('Group of samples')
plt.ylabel('Phase (radians/pi)')
plt.title('Intercept of least-squares regression lines of unwrapped phases')