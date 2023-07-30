# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:43:17 2022

@author: lisow
"""

import os
# import librosa
# import matplotlib.pyplot as plt
import evaluate
import csv
import utils
import numpy as np
import scipy.signal
import plots2


boundary_sec = np.array([0, 19, 38, 57, 75, 92, 106, 124, 142, 159, 177, 194, 212, 230, 247])
rep_sec = np.array([[35,38], [52,57], [72,75], [89,93], [103,106], [119,124], [151,154], [173,178], [192,196], [208,213], [244,248]])
mext_sec = np.array([1, 2, 3, 20, 28, 45.5, 88, 89, 114.5, 184.5, 235.8]) #

boundary_frame = boundary_sec*5 #sr/float(hop_length)
rep_frame = rep_sec*5
mext_frame = mext_sec*5

############# C-M ################

crepe_file = 'crepe.csv'
madmom_file = 'OnsetDetector.csv'
midi_file = 'result_revision/DEBASHISH_SANYAL_1b_lam0.001_tran_3_raga_1.mid'
# Note_all = evaluate.get_crepe_madmom(crepe_file, madmom_file, midi_file)
Note_all = utils.midi_to_notes(midi_file)
# metric_position, note_value = notetrans.note_to_note_value(beats[0], Note_all['coarse'])


# SSM_r7 = utils.rhythm_SSM(Note_all['coarse'], note_value, 7)   
S = utils.local_align_SSM(Note_all, 7, 0, 0)
# SSM_7 = SSM_r7*SSM_p7

rep_ = utils.repetition(S)
struct_bound_ = utils.struct_boundary(S)
melody_ext_, ext_time = utils.melody_extension(Note_all, 0)


bdy_idx = scipy.signal.argrelextrema(struct_bound_, np.greater)
rep_idx = scipy.signal.argrelextrema(rep_, np.greater)
mext_idx = scipy.signal.argrelextrema(melody_ext_, np.greater)

result_ana = evaluate.evaluation_ana(bdy_idx[0], rep_idx[0], mext_idx[0], boundary_frame, rep_frame, mext_frame)

savename = 'test2.png'
plotfig = plots2.plots2()
plotfig.plot_fig7(S, struct_bound_, rep_, ext_time, savename)
# baseline_midi_path = 'ISMIR2014_dataset_result/tony'
# gt_path = 'ISMIR2014_dataset_result/gt'

# baseline_list = os.listdir(baseline_midi_path)
# gt_list = os.listdir(gt_path)

# with open('tony_ismir2014.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)    
#     for i in range(len(baseline_list)):
#         baseline = baseline_midi_path + '/' + baseline_list[i]
#         gt = gt_path + '/' + gt_list[i]
#         Acc, Pseq, Rseq, F1seq = evaluate.evaluate_transcription(gt, baseline, 0)
#         writer.writerow([Acc, Pseq, Rseq, F1seq])

# SF_maxfilt_bw = 61
# SF_thres = 0.1
# lam = 1E-3
# Note_th = 5
# Repeated_note_th = 15 
# Transition_th = 3
# note_level = cfp_hmlc.Note_level_processing(SF_maxfilt_bw, SF_thres, lam, Note_th, Repeated_note_th, Transition_th)

##########################

# gt_filename = 'music_sitar/Debashish Sitar Solo 2.mid'
# result_filename = 'tony.mid'   
# Notes = evaluate.tony_processing(result_filename)
# Notes = utils.remove_outlier(Notes)
# utils.note_to_midi(Notes, 'tony_processed.mid', 1)
# Acc, Pseq, Rseq, F1seq = evaluate.evaluate_transcription(gt_filename, 'tony_processed.mid', 1)
