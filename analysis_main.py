# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:43:17 2022

@author: Li Su

This demo code is to compute the structure boundary, 
repeated patterns (tihais), and melody extension. 
The results are also evaluated against the ground 
truth annotation and are illustrated.

The MIDI file outputted from the AMT model is used
here for the analysis tasks. 
"""

import evaluate
import utils
import numpy as np
import scipy.signal
import plots2


########### Ground truth labels (all are in seconds)
"""
The ground truth labels of section boundary, repeated 
note sequence sectopms (tihais) and melody extensions. 
All the labels are in seconds.
"""
boundary_sec = np.array([0, 19, 38, 57, 75, 92, 106, 124, 142, 159, 177, 194, 212, 230, 247])
rep_sec = np.array([[35,38], [52,57], [72,75], [89,93], [103,106], [119,124], [151,154], [173,178], [192,196], [208,213], [244,248]])
mext_sec = np.array([1, 2, 3, 20, 28, 45.5, 88, 89, 114.5, 184.5, 235.8]) #

boundary_frame = boundary_sec*5 #sr/float(hop_length)
rep_frame = rep_sec*5
mext_frame = mext_sec*5

############# analysis

seq_len = 7 # the length of sequence for computation of the LCS distance

# specify the midi file transcribed from the AMT model
midi_file = 'result/DEBASHISH_SANYAL_1b_20221205.mid'
Note_all = utils.midi_to_notes(midi_file)
  
S = utils.local_align_SSM(Note_all, seq_len, 0, 0)

rep_ = utils.repetition(S)
struct_bound_ = utils.struct_boundary(S)
melody_ext_, ext_time = utils.melody_extension(Note_all, 0)


bdy_idx = scipy.signal.argrelextrema(struct_bound_, np.greater)
rep_idx = scipy.signal.argrelextrema(rep_, np.greater)
mext_idx = scipy.signal.argrelextrema(melody_ext_, np.greater)

############# evaluation

result_ana = evaluate.evaluation_ana(bdy_idx[0], rep_idx[0], mext_idx[0], boundary_frame, rep_frame, mext_frame)

############ plot the SSM and analysis result

figname = 'HMLC_CFP.png' # specify the name of the figure here
plotfig = plots2.plots2()
plotfig.plot_fig7(S, struct_bound_, rep_, ext_time, figname)
