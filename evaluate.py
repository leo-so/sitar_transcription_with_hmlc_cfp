# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 09:53:31 2020

@author: lisu
"""

# import editdistance
import numpy as np
import pretty_midi
import alignment
import csv
import utils
import cfp_hmlc
# import mir_eval

### processing of baseline methods

def freq2midi(f):
    return 69.0+12*np.log2(f/440.0)

def freq2cent(f):
    return 60*np.log2(f/130.81)

def get_crepe_madmom(crepe_file, madmom_file, midi_filename):
    SF_maxfilt_bw = 61
    SF_thres = 0.1
    lam = 1E-3
    Note_th = 5
    Repeated_note_th = 15 
    Transition_th = 3
    note_level = cfp_hmlc.Note_level_processing(SF_maxfilt_bw, SF_thres, lam, Note_th, Repeated_note_th, Transition_th)
    
    frame_pitch = []
    onset = []
    Note_all = []
    with open(crepe_file, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if not row[1] == 'frequency':
                frame_pitch.append(freq2cent(float(row[1])))
                # print(freq2midi(float(row[1])))
    
    with open(madmom_file) as csvfile: #, new_line=''
        rows = csv.reader(csvfile)
        for row in rows:
            onset.append(int(float(row[0])*100))
    
    Note_all = note_level.post_proc(frame_pitch, frame_pitch, onset)    

    Note_all['coarse'] = utils.remove_outlier(Note_all['coarse'])
    # utils.note_to_midi(Note_all['coarse'], midi_filename, 1)
    return Note_all


def get_tony(filename):
#    gt_midi = pretty_midi.PrettyMIDI('music_sitar/Debashish Sitar Solo 2.mid')
    gt_midi = pretty_midi.PrettyMIDI(filename) #('midi/crepe_madmom2.mid')
    gt_notes=[]

    for i in range(len(gt_midi.instruments)):
        for j in range(len(gt_midi.instruments[i].notes)):
            note_start = int(gt_midi.instruments[i].notes[j].start*100)
            note_end = int(gt_midi.instruments[i].notes[j].end*100)
            pitch = 5*(gt_midi.instruments[i].notes[j].pitch-48)
            No ={'pitch': np.array([[pitch]]),\
                 'time': np.array([np.arange(note_start,note_end+1)])}
            gt_notes.append(No)
    return gt_notes

### evaluation metrics

def get_lcs(gt_notes, est_notes):
    len1 = len(gt_notes)
    len2 = len(est_notes)
    lcs = []
    for i in range(len1):
        for j in range(len2):
            lcs_temp = 0
            match = []
            while ((i+lcs_temp < len1) and (j+lcs_temp<len2) and gt_notes[i+lcs_temp] == est_notes[j+lcs_temp]):
                match.append(est_notes[j+lcs_temp])
                lcs_temp+=1
            if (len(match) > len(lcs)):
                lcs = match
    return lcs, len(lcs)

# def get_editdistance(gt_notes, est_notes):
#     editD = editdistance.eval(gt_notes, est_notes) 
#     return editD

### evaluation

def evaluate_transcription(gt_filename, result_filename, calibration):
    # gtname = 'music_sitar/Debashish Sitar Solo 2.mid'
    # resultname = 'N_5_H_3_t.mid' 
    # raga_list = np.asarray([48, 50, 53, 55, 58, 60, 62, 65, 67, 70, 72, 74])
    # one beat = 0.5454545 second (110 BPM)
        
    gt_midi = pretty_midi.PrettyMIDI(gt_filename)
    est_midi = pretty_midi.PrettyMIDI(result_filename)
    gt_notes=[]
    est_notes=[]

    for i in range(len(gt_midi.instruments)):
        for j in range(len(gt_midi.instruments[i].notes)):
            gt_notes.append(gt_midi.instruments[i].notes[j].pitch + calibration)
    
    for i in range(len(est_midi.instruments[0].notes)):
        note_number = est_midi.instruments[0].notes[i].pitch
        # note_number = to_raga_note(note_number)
        est_notes.append(note_number)
    
    # A, score, best, opt_loc = local_align(gt_notes, est_notes, -1, 100, -0.99)
    X, Y, opt_loc = alignment.local_align(gt_notes, est_notes, -1, 100, -0.99)

    gt_notes=np.array(gt_notes)
    est_notes=np.array(est_notes)
        
    # normalized graph match
    Pseq = len(opt_loc)/float(est_notes.shape[0]) #float(opt_loc.shape[0])/float(est_notes.shape[0])
    Rseq = len(opt_loc)/float(gt_notes.shape[0]) #float(opt_loc.shape[0])/float(gt_notes.shape[0])
    F1seq = 2*Pseq*Rseq/(Pseq+Rseq)
    Acc = len(opt_loc)/(float(est_notes.shape[0])+float(gt_notes.shape[0])-len(opt_loc))
        
    return Acc, Pseq, Rseq, F1seq


def evaluate_note_value(gt_filename, result_filename, note_value):
    # gtname = 'music_sitar/Debashish Sitar Solo 2.mid'
    # resultname = 'N_5_H_3_t.mid' 
    # raga_list = np.asarray([48, 50, 53, 55, 58, 60, 62, 65, 67, 70, 72, 74])
    # one beat = 0.5454545 second (110 BPM)
        
    gt_midi = pretty_midi.PrettyMIDI(gt_filename)
    est_midi = pretty_midi.PrettyMIDI(result_filename)
    gt_notes=[]
    est_notes=[]
    gt_val=[]

    for i in range(len(gt_midi.instruments)):
        for j in range(len(gt_midi.instruments[i].notes)):
            gt_notes.append(gt_midi.instruments[i].notes[j].pitch+1)
            val = int(round(4*(gt_midi.instruments[i].notes[j].end-gt_midi.instruments[i].notes[j].start)/0.5454545))
            gt_val.append(val)
    
    for i in range(len(est_midi.instruments[0].notes)):
        note_number = est_midi.instruments[0].notes[i].pitch
        # note_number = to_raga_note(note_number)
        est_notes.append(note_number)

    note_value_ = 4*note_value
    note_value_.astype('int')
    X, Y, opt_loc = alignment.local_align(gt_val, note_value_, -1, 100, -0.99)

    gt_notes=np.array(gt_notes)
    est_notes=np.array(est_notes)
        
    # normalized graph match
    Pseq = len(opt_loc)/float(est_notes.shape[0]) #float(opt_loc.shape[0])/float(est_notes.shape[0])
    Rseq = len(opt_loc)/float(gt_notes.shape[0]) #float(opt_loc.shape[0])/float(gt_notes.shape[0])
    F1seq = 2*Pseq*Rseq/(Pseq+Rseq)
    Acc = len(opt_loc)/(float(est_notes.shape[0])+float(gt_notes.shape[0])-len(opt_loc))
        
    return Acc, Pseq, Rseq, F1seq

def evaluation_ana(bdy, rep, mext, bdy_gt, rep_gt, mext_gt):
    bdy_TP = 0.0
    for i in range(bdy.shape[0]):
        for j in range(bdy_gt.shape[0]):
            if abs(bdy[i]-bdy_gt[j])<=15: # 3 secs
                bdy_TP = bdy_TP +1
    bdy_P = bdy_TP/bdy.shape[0]
    bdy_R = bdy_TP/bdy_gt.shape[0]
    bdy_F = 2*(bdy_P*bdy_R)/(bdy_P+bdy_R)
    
    rep_TP = 0.0
    for i in range(rep.shape[0]):
        for j in range(rep_gt.shape[0]):
            if rep[i]>=rep_gt[j,0] and rep[i]<=rep_gt[j,1]:
                rep_TP = rep_TP +1
    rep_P = rep_TP/rep.shape[0]
    rep_R = rep_TP/rep_gt.shape[0]
    rep_F = 2*(rep_P*rep_R)/(rep_P+rep_R)    
    
    mext_TP = 0.0
    for i in range(mext.shape[0]):
        for j in range(mext_gt.shape[0]):
            if abs(mext[i]-mext_gt[j])<=5: # 1 sec
                mext_TP = mext_TP +1
    mext_P = mext_TP/mext.shape[0]
    mext_R = mext_TP/mext_gt.shape[0]
    mext_F = 2*(mext_P*mext_R)/(mext_P+mext_R)
    
    result_ana = [bdy_P, bdy_R, bdy_F, rep_P, rep_R, rep_F, mext_P, mext_R, mext_F]
    
    return result_ana

# if __name__== "__main__":
#     ind = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
#     MulRes = 0
#     TempoEn = 1
#     Har = 3
#     Result = np.zeros((14,5))
    
#     for fileindex in range(0,1): #range(5,14):
# #        gtname = 'music_sitar/sitar_' + ind[fileindex] + '.mid'
#         gtname = 'music_sitar/Debashish Sitar Solo 2.mid'
# #        resultname = 'coarse_new_sitar_' + ind[fileindex] + '_MulRes_' + str(MulRes) + '_TempoEn_' + str(TempoEn) + '_Har_' + str(Har) + '.mid'
# #        resultname = 'midi/crepe_madmom2.mid' #
# #        resultname = 'coarse_sitar_' + '_MulRes_' + str(MulRes) + '_TempoEn_' + str(TempoEn) + '_Har_' + str(Har) + '.mid' #
# #        resultname = 'tony.mid' 
#         resultname = 'N_5_H_3_t.mid' 

#         raga_list = np.asarray([48, 50, 53, 55, 58, 60, 62, 65, 67, 70, 72, 74])
#         # one beat = 0.5454545 second (110 BPM)
        
#         gt_midi = pretty_midi.PrettyMIDI(gtname)
#         est_midi = pretty_midi.PrettyMIDI(resultname)
#         gt_notes=[]
#         est_notes=[]
#         gt_val=[]

#         for i in range(len(gt_midi.instruments)):
#             for j in range(len(gt_midi.instruments[i].notes)):
#                 gt_notes.append(gt_midi.instruments[i].notes[j].pitch+1)
#                 val = int(round(4*(gt_midi.instruments[i].notes[j].end-gt_midi.instruments[i].notes[j].start)/0.5454545))
#                 gt_val.append(val)
    
#         for i in range(len(est_midi.instruments[0].notes)):
#             note_number = est_midi.instruments[0].notes[i].pitch
#             note_number = raga_list[np.where(abs(note_number-raga_list)==np.amin(abs(note_number-raga_list)))][0]
#             est_notes.append(note_number)
    
#         editD = editdistance.eval(gt_notes, est_notes)   
#         _, llcs = get_lcs(gt_notes, est_notes)
# #        A, score, best, opt_loc = local_align(gt_notes, est_notes, -1, 100, -0.99)
# #        X, Y, opt_loc = alignment.local_align(gt_notes, est_notes, -1, 100, -0.99)
#         note_value_ = 4*note_value
#         note_value_.astype('int')
#         X, Y, opt_loc = alignment.local_align(gt_val, note_value_, -1, 100, -0.99)

#         gt_notes=np.array(gt_notes)
#         est_notes=np.array(est_notes)
        
#         # normalized edit distnace
#         NeditD = 1-float(editD)/float(gt_notes.shape[0]+est_notes.shape[0])
#         # Substring-level P, R, F1
#         Pstr = float(llcs)/float(est_notes.shape[0])
#         Rstr = float(llcs)/float(gt_notes.shape[0])
#         F1str = 2*Pstr*Rstr/(Pstr+Rstr)
#         # normalized graph match
#         Pseq = len(opt_loc)/float(est_notes.shape[0]) #float(opt_loc.shape[0])/float(est_notes.shape[0])
#         Rseq = len(opt_loc)/float(gt_notes.shape[0]) #float(opt_loc.shape[0])/float(gt_notes.shape[0])
#         F1seq = 2*Pseq*Rseq/(Pseq+Rseq)
#         Acc = len(opt_loc)/(float(est_notes.shape[0])+float(gt_notes.shape[0])-len(opt_loc))
        
    
#         Result[fileindex,0] = Acc
#         Result[fileindex,1] = Pseq
#         Result[fileindex,2] = Rseq
#         Result[fileindex,3] = F1seq
#         Result[fileindex,4] = 0 

    
#    matches = mir_eval.transcription.match_notes(gt_intervals, gt_notes, est_intervals, est_notes, onset_tolerance=1, pitch_tolerance=50.0, offset_ratio=10000, offset_min_tolerance=10000)

    
