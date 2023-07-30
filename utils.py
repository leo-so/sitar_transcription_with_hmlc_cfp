# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:00:16 2022

@author: Li Su
"""

import numpy as np
import pretty_midi
import alignment
from scipy.interpolate import griddata
from sklearn.metrics import pairwise_distances
import scipy.signal

mode = 'all'

def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g!=0:
        X[X<0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X

def DPNT(X, lam):
    X = np.hstack((X[:,0:1], X))
    M, N = X.shape
    contour = np.zeros(N)
    contour_val = np.zeros(N)
    C = np.zeros((M,N))
    FVal = -np.inf*np.ones((M,N)) # M is freq, N is time
    FVal[:,0] = X[:,0]
    PreIdx = np.zeros((M,N))
    PreIdx[:,0] = np.argmax(X[:,0])*np.ones(M)

    for ti in range(1,N):
        for fi in range(0,M):
            temp = FVal[fi,ti-1]*np.ones(M)+X[:,ti]-lam*(np.arange(0,M)-fi*np.ones(M))**2 #np.maximum((np.arange(0,M)-fi*np.ones(M))**2-1,0) #
            temp[:max(0,fi-20)]=-np.inf
            temp[min(fi+20,M-1):]=-np.inf
            FVal[fi,ti] = np.amax(temp)
            PreIdx[fi,ti] = np.argmax(temp)    

    for tii in range(N-1,-1,-1):
        if tii == N-1:
            P = int(PreIdx[np.argmax(FVal[:,tii]), tii])
            C[P, tii] = 1
            contour[tii] = P
            contour_val[tii] = X[P,tii]
        else:
            P = int(PreIdx[P,tii])
            C[P, tii] = 1
            contour[tii] = P
            contour_val[tii] = X[P,tii]
    C = C[:, 1:]
    contour = contour[1:]
    contour_val = contour_val[1:]
    return C, contour, contour_val

def findpeaks_time(Z, th): 
    M = Z.shape[0]    
    pre = Z[1:M-1] - Z[0:M-2]
    pre[pre<0] = 0
    pre[pre>0] = 1
    post = Z[1:M-1] - Z[2:]
    post[post<0] = 0
    post[post>0] = 1

    mask = pre*post
    ext_mask = np.append([0], mask)
    ext_mask = np.append(ext_mask, [0])
    
    pdata = Z * ext_mask
    pdata = pdata-th*np.amax(pdata, axis=0) 
    pdata[pdata<0] = 0
    pdata = pdata.reshape(Z.shape)
    
    locs = np.where(pdata>0)
    return pdata, locs

def to_raga_note(note_number):
    raga_list = np.asarray([48, 50, 53, 55, 58, 60, 62, 65, 67, 70, 72, 74])
    # print(np.where(abs(note_number-raga_list)==np.amin(abs(note_number-raga_list))))
    note_number = raga_list[np.where(abs(note_number-raga_list)==np.amin(abs(note_number-raga_list)))][0]
    return note_number

def note_to_midi(Note, filename, raga):
    # 130.81 Hz = midi number 48
    # raga_list = np.asarray([48, 50, 53, 55, 58, 60, 62, 65, 67, 70, 72, 74])
    
    # Create a PrettyMIDI object
    sitar_transcription = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    sitar_program = pretty_midi.instrument_name_to_program('Acoustic Guitar (steel) ')
    sitar = pretty_midi.Instrument(program=sitar_program)
    
    NumOfNote = len(Note)
    for j in range(NumOfNote):
        # Retrieve the MIDI note number for this note name
        note_number = int(np.round(48+Note[j]['pitch'][0,0]/5.0)) #int(np.round(48+Note[j]['pitch'][0,0]/5.0))#int(12.0*np.log2(note_freq/440.0)+69.0)
        if raga == 1:
            note_number = to_raga_note(note_number)
        # note_number = raga_list[np.where(abs(note_number-raga_list)==np.amin(abs(note_number-raga_list)))][0]
        start_time = Note[j]['time'][0,0]*0.01
        end_time = Note[j]['time'][0,-1]*0.01
        # Create a Note instance for this note, starting at 0s and ending at wh.5s
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=end_time)
        sitar.notes.append(note)
        
    # Add the instrument to the PrettyMIDI object
    sitar_transcription.instruments.append(sitar)
    # Write out the MIDI data
    sitar_transcription.write(filename)

def note_to_midi_baseline(Note, filename):
    # 130.81 Hz = midi number 48
    # raga_list = np.asarray([48, 50, 53, 55, 58, 60, 62, 65, 67, 70, 72, 74])
    
    # Create a PrettyMIDI object
    sitar_transcription = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    sitar_program = pretty_midi.instrument_name_to_program('Acoustic Guitar (steel) ')
    sitar = pretty_midi.Instrument(program=sitar_program)
    
    NumOfNote = len(Note)
    for j in range(NumOfNote):
        # Retrieve the MIDI note number for this note name
        note_number = int(Note[j]['pitch'][0,0]) #int(np.round(48+Note[j]['pitch'][0,0]/5.0))#int(12.0*np.log2(note_freq/440.0)+69.0)
        # note_number = raga_list[np.where(abs(note_number-raga_list)==np.amin(abs(note_number-raga_list)))][0]
        note_number = to_raga_note(note_number)
        start_time = Note[j]['time'][0,0]*0.01
        end_time = Note[j]['time'][0,-1]*0.01
        # Create a Note instance for this note, starting at 0s and ending at wh.5s
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=end_time)
        sitar.notes.append(note)
        
    # Add the instrument to the PrettyMIDI object
    sitar_transcription.instruments.append(sitar)
    # Write out the MIDI data
    sitar_transcription.write(filename)

def midi_to_notes(filename):
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

def local_align_SSM(Note, n, mode, interval):
    raga_list = np.array([0, 9, 24, 34, 49, 59, 69, 84, 94, 109, 119, 129])
    N = (n-1)/2.0
    NumOfNote = len(Note)
    pitch_seq = []
    onset_time = []
    max_time = Note[NumOfNote-1]['time'][0,-1]
    print(max_time)
    for i in range(NumOfNote):
        seg = []
        for j in range(max(i-int(np.floor(N)),0), min(i+int(np.ceil(N))+1,NumOfNote)):
            if mode == 'GT':
                note_ = Note[j]['pitch'][0,0]
            else:
                note_ = raga_list[np.where(abs(Note[j]['pitch'][0,0]-raga_list)== \
                                np.amin(abs(Note[j]['pitch'][0,0]-raga_list)))[0][0]]            
            seg.append(note_)
            
        if interval == 1:        
            seg = np.array(seg)
            seg = seg[1:]-seg[:-1]
            seg = seg.tolist()
        
        pitch_seq.append(seg)
        onset_time.append(Note[i]['time'][0,0])
    
    NumOfSeq = len(pitch_seq)    
    SSM = np.zeros((NumOfSeq, NumOfSeq))
    P_x = np.zeros((NumOfSeq, NumOfSeq))
    P_y = np.zeros((NumOfSeq, NumOfSeq))
    for i in range(NumOfSeq):
        for j in range(NumOfSeq):
            _, _, opt_loc = alignment.local_align(pitch_seq[i], pitch_seq[j], -1, 10, -0.99)
            SSM[i,j] = len(opt_loc) #.shape[0]
            P_x[i,j] = onset_time[i]
            P_y[i,j] = onset_time[j]
            
    points = np.hstack((np.reshape(P_x,[-1,1]),np.reshape(P_y,[-1,1])))
    values = np.reshape(SSM,[-1,1])
#    grid_x, grid_y = np.mgrid[0:max_time:5, 0:max_time:5] # resolution = 0.01sec
    grid_x, grid_y = np.mgrid[0:max_time:20, 0:max_time:20] # resolution = 0.2sec
    SSM_grid = griddata(points, values, (grid_x, grid_y), method='nearest')
    return SSM_grid[:,:,0] #SSM #

def rhythm_SSM(Note, note_value, n):
    N = (n-1)/2.0
    NumOfNote = note_value.shape[0]
    nv_seq = []
    onset_time = []
    max_time = Note[NumOfNote-1]['time'][0,-1]
    for i in range(NumOfNote):
        seg = []
        for j in range(max(i-int(np.floor(N)),0), min(i+int(np.ceil(N))+1,NumOfNote)):          
            seg.append(note_value[j])        
        nv_seq.append(seg)
        onset_time.append(Note[i]['time'][0,0])
    
    NumOfSeq = len(nv_seq)    
    SSM = np.zeros((NumOfSeq, NumOfSeq))
    P_x = np.zeros((NumOfSeq, NumOfSeq))
    P_y = np.zeros((NumOfSeq, NumOfSeq))
    for i in range(NumOfSeq):
        for j in range(NumOfSeq):
            _, _, opt_loc = alignment.local_align(nv_seq[i], nv_seq[j], -1, 10, -0.99)
            SSM[i,j] = len(opt_loc) #.shape[0]
            P_x[i,j] = onset_time[i]
            P_y[i,j] = onset_time[j]
            
    points = np.hstack((np.reshape(P_x,[-1,1]),np.reshape(P_y,[-1,1])))
    values = np.reshape(SSM,[-1,1])
    grid_x, grid_y = np.mgrid[0:max_time:20, 0:max_time:20] # resolution = 0.2sec
    SSM_grid = griddata(points, values, (grid_x, grid_y), method='nearest')
    return SSM_grid[:,:,0] #SSM #

# audio ssm
def audio_SSM(Z): # frame rate of Z: 100 Hz
    Z_ = Z[:, 20::20]  # frame rate to 5 Hz (0.2 sec)
    for i in range(1,11):
        Z_ = np.vstack((Z_, Z[:, 20-2*i:-2*i:20]))
    R = 1-pairwise_distances(Z_.T, metric="cosine")
    return R

#   short-term repetition (find tihais)
def repetition(SSM):
    sw = 20 # The search range (search width) to find repetition in the SSM        
    qfactor = np.zeros(SSM.shape[0]) 
    for i in range(1,int(SSM.shape[0]-sw/2-1)):
        trigger = 0
        num_of_peaks = 0
        for j in range(i+1, min(i+sw, SSM.shape[0]-1)): #max(i-sw,1)
            if SSM[i,j]>SSM[i,j-1] and SSM[i,j]>=np.amax(SSM)/2.0:
                trigger = 1
            elif SSM[i,j]<SSM[i,j-1] and trigger == 1:
                num_of_peaks = num_of_peaks + 1
                trigger = 0
        qfactor[int(i+sw/2)]= np.mean(SSM[i,i+1:min(i+sw, SSM.shape[0]-1)])*(num_of_peaks) # max(i-sw,1)#        

    qfactor = np.convolve(qfactor, scipy.signal.hanning(sw*2)/sw, mode='same')
    qfactor = qfactor/np.amax(qfactor)
    
    return qfactor
 
# structure boundary with checkboard filter
def struct_boundary(SSM):
    sw = 40 # search width to find boundaries in the SSM
    M = SSM.shape[0]
    boundary = np.zeros(M)
    for i in range(7,M-7):      
        boundary[i] = np.mean(SSM[max(i-sw,0):i,max(i-sw,0):i])+np.mean(SSM[i+1:min(i+sw,M),i+1:min(i+sw,M)])-\
                    np.mean(SSM[max(i-sw,0):i,i+1:min(i+sw,M)])-np.mean(SSM[i+1:min(i+sw,M),max(i-sw,0):i])

    boundary = np.convolve(boundary, scipy.signal.hanning(sw)/sw, mode='same')
    boundary = boundary/np.amax(boundary)    
    print(boundary.shape)
    print(boundary)
    return boundary

# melody extension
def melody_extension(Note, gt_mode):
    raga_list = np.array([0, 9, 24, 34, 49, 59, 69, 84, 94, 109, 119, 129])
    note_seq = []
    max_time = Note[len(Note)-1]['time'][0,-1]
#    if mode == 'all':
    melody_ext = np.zeros(int(max_time/20))
#    else:
#        melody_ext = np.zeros(int(max_time))
    for i in range(len(Note)):
        if gt_mode == 'GT':
#            note_ = Note[i]['pitch'][0,0]
            note_ = raga_list[np.where(abs(Note[i]['pitch'][0,0]-raga_list)== \
                        np.amin(abs(Note[i]['pitch'][0,0]-raga_list)))[0][0]]
        else:
            note_ = raga_list[np.where(abs(Note[i]['pitch'][0,0]-raga_list)== \
                        np.amin(abs(Note[i]['pitch'][0,0]-raga_list)))[0][0]]
        if note_ not in note_seq:
            if mode == 'all':
                t = int(Note[i]['time'][0,0]/20.0)
            else:
                t = int(Note[i]['time'][0,0])
            melody_ext[t] = 0.5                
            note_seq.append(note_)
    ext_time = np.where(melody_ext==0.5)[0]
    return melody_ext, ext_time

def melody_extension_audio(SSM):
    melody_ext = np.zeros(SSM.shape[0])
    melody_ext[0] = 1
    for i in range(1,SSM.shape[0]):
        melody_ext[i] = (SSM[i,i]-np.amax(SSM[i,:i]))
        
    melody_ext = np.convolve(melody_ext, scipy.signal.hanning(20), mode='same')
    melody_ext = melody_ext/np.amax(melody_ext)
    return melody_ext

def remove_outlier(Note_):  
# raga notes should be longer than non-rage notes
# repeated notes / non-repeated notes
# drone notes / played notes      
    Note = list(Note_)
# remove outlier by note sequence median value
    N = len(Note)
    R_idx = []
    pitch_vec_all = np.zeros((1,0))
    for i in range(N):
        pitch_vec_all = np.append(pitch_vec_all, Note[i]['pitch'][0,0])
    pitch_vec_all = np.append(pitch_vec_all[0]*np.ones(6), pitch_vec_all) #np.reshape(pitch_vec_all,[1,-1])
    pitch_vec_median = scipy.signal.medfilt(pitch_vec_all, 13)
    pitch_vec_median = pitch_vec_median[6:]
#    print(pitch_vec_median)
    for i in range(N):
        if abs(Note[i]['pitch'][0,0]-pitch_vec_median[i])>45:
            R_idx.append(i)
    for index in sorted(R_idx, reverse=True):
        del Note[index]
        
    # remove non-raga notes
#    raga_list = np.array([0, 9, 24, 34, 49, 59, 69, 84, 94, 109, 119, 129])
#    N = len(Note)
#    R_idx = []
#    for i in range(N):
#        if min(np.abs(Note[i]['pitch'][0,0]-raga_list))>=5:
#            R_idx.append(i)
#    for index in sorted(R_idx, reverse=True):
#        del Note[index]
        
    # remove short notes 
    N = len(Note)
    R_idx = []
    for i in range(N):
        if Note[i]["pitch"][0,0]<=5:
            R_idx.append(i)
    for index in sorted(R_idx, reverse=True):
        del Note[index]
    
    # remove the unual pitch contour
#    N = len(Note)
#    R_idx = []
#    for i in range(N):
#        if np.amax(np.abs(np.diff(Note[i]['contour_fine'],axis=1)))>=8:
#            R_idx.append(i)
#    for index in sorted(R_idx, reverse=True):
#        del Note[index]
        
    # remove the fluctuated notes by IF contour
#    N = len(Note)
#    R_idx = []
#    for i in range(N):
#        print(fluc_rate(Note[i]["contour_fine"]))
#        if fluc_rate(Note[i]["contour_fine"])>20:            
#            R_idx.append(i)
#    for index in sorted(R_idx, reverse=True):
#        del Note[index]

#     remove the outlier notes by SSM
#    SSM = local_align_SSM(Note, 5)
#    SSM[SSM>0] = 1.0
#    similarity_curve = np.sum(SSM, axis = 0)/SSM.shape[0]
##    print(np.median(SSM))
#    R_idx = np.where(similarity_curve < 0.2*np.amax(similarity_curve))[0]
#    for index in sorted(R_idx, reverse=True):
#        del Note[index]
    
    # remove the outlier notes by n-gram
#    N = len(Note)
#    R_idx = []
#    for i in range(N):
#        feat = [abs(Note[i]["pitch_fine"][0,0] - Note[max(i-3,0)]["pitch_fine"][0,0]), \
#                abs(Note[i]["pitch_fine"][0,0] - Note[max(i-2,0)]["pitch_fine"][0,0]), \
#                abs(Note[i]["pitch_fine"][0,0] - Note[max(i-1,0)]["pitch_fine"][0,0]), \
#                abs(Note[i]["pitch_fine"][0,0] - Note[min(i+1,N-1)]["pitch_fine"][0,0]), \
#                abs(Note[i]["pitch_fine"][0,0] - Note[min(i+2,N-1)]["pitch_fine"][0,0]), \
#                abs(Note[i]["pitch_fine"][0,0] - Note[min(i+3,N-1)]["pitch_fine"][0,0])] 
#        feat = np.asarray(feat)
#        if len(np.where(feat>30)[0])>=2:
#            R_idx.append(i)
#    for index in sorted(R_idx, reverse=True):
#        del Note[index]

    Note_new = Note
    return Note_new #, outlier_ind