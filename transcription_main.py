# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 23:23:15 2022

@author: Li Su

This is a demo code for CFP-HMLC, an automatic music 
transcription (AMT) method which is purely based on 
signal processing. Suggested parameters are listed as 
follows.

The method take a .wav file as input and output the 
transcribed results in the MIDI format.

To run the demo code, please specify the input path 
(for the .wav file) and the output path which is to 
save the output MIDI file. If you are to run the 
evaluation, also specify the path of the ground 
truth MIDI.
"""

import numpy as np
import cfp_hmlc
import utils
import evaluate

if __name__== "__main__":
    
    # create a CFP-HMLC object for pitch transcription
    fs = 16000 # audio sampling rate in Hz
    win = 2829 # window size in samples
    hop = 160 # hop size in samples
    fr = 1.0 # frequency resolution in Hz (the spacing of FFT bins)
    fc = 130.81 # cutoff frequency (i.e., the lowest frequency considered)
    tc = 1/(fc*32.0) # cutoff quefrency (i.e., the highest frequency considered)
    g = np.array([0.2, 0.6, 0.9, 0.9, 1.0]) # the gamma parameter of the generalized cepstrum
    NumPerOctave = 60 # number of bins per octave after the filterbank
    bov = 2 # overlap factor of the filterbank
    Har = 3 # number of harmonics to be computed
    cfp_hmlc_pitch = cfp_hmlc.CFP_HMLC(fs, win, hop, fr, fc, tc, g, NumPerOctave, bov, Har)
    
    # create a CFP-HMLC object for tempo estimation
    fs = 100.0 # frame rates (the sampling rate of the onset detection curve)
    win = 801 
    hop = 20 
    fr = 0.01 #fs/SF.shape[0]
    fc = 0.7
    tc = 1.0/3.0
    g = np.array([0.2, 0.6, 1.0]) 
    NumPerOctave = 60 
    bov = 2 
    Har = 1
    cfp_hmlc_tempo = cfp_hmlc.CFP_HMLC(fs, win, hop, fr, fc, tc, g, NumPerOctave, bov, Har)

######### parameters for note transcription     
    SF_maxfilt_bw = 61 # bandwidth of the maximal filter in computing onsets in bins
    SF_thres = 0.1 # threshold of onset detection
    lam = 1E-3 # parameter of the DP-based pitch contour tracking
    Note_th = 5 # threshold of note length, in frames
    Repeated_note_th = 15 # threshold of repeated note segmentation, in frames
    Transition_th = 3 # threshold of note transition
    note_level = cfp_hmlc.Note_level_processing(SF_maxfilt_bw, SF_thres, lam, Note_th, Repeated_note_th, Transition_th)
    tempo_beat = cfp_hmlc.CFP_HMLC_tempo_beat(cfp_hmlc_tempo)
    notetrans = cfp_hmlc.note_transcription(cfp_hmlc_pitch, note_level)

######### input file path and save path
    inputname = 'DEBASHISH_SANYAL_1b.wav' # specify the input file here
    outputname = 'DEBASHISH_SANYAL_1b_20221205.mid' # specify the output filename here
    
######### compute CFP feature (Z) and note list            
######### tempo, beats, PLP curve, the CFP-HMLC of the novelty curve, and     
    Note_all, Z, S, SF, CenFreq = notetrans.transcribe_note(inputname)
    tempo, beats, PLP, TP, ffff = tempo_beat.estimate_tempo_beat(SF)
    metric_position, note_value = notetrans.note_to_note_value(beats[0], Note_all['coarse'])

####### export to midi    
    utils.note_to_midi(Note_all['coarse'], outputname, 1)

####### evluation
    gt_filename = 'case_study_sample/Debashish Sitar Solo 2.mid'
    result_filename = outputname 
    Acc, Pseq, Rseq, F1seq = evaluate.evaluate_transcription(gt_filename, result_filename, 1)


