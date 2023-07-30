# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:20:54 2022

@author: lisow
"""

import numpy as np
import scipy.signal
import scipy.ndimage
import scipy.fftpack
import utils
import soundfile as sf

class CFP_HMLC:
    def __init__(self, fs, win, hop, fr, fc, tc, g, NumPerOctave, bov, Har):
        self.fs = fs #16000.0
        self.win = win # 2829
        self.hop = hop
        self.fr = fr
        self.fc = fc
        self.tc = tc
        self.g = g
        self.NumPerOct = NumPerOctave
        self.bov = bov
        self.Har = Har
        # fr = 1.0 #0.5 # frequency resolution
        # fc = 130.81 # the frequency of the lowest pitch
        # tc = 1/(fc*32.0) # the period of the highest pitch
        # g = np.array([0.2, 0.6, 0.9, 0.9, 1])
        # NumPerOctave = 60 # Number of bins per octave
        # bov = 2
        
    def get_window(self):
        return scipy.signal.blackmanharris(self.win)

    def STFT_complex(self, x, h):      
        # h = self.get_window(self) 
        t = np.arange(self.hop, np.ceil(len(x)/float(self.hop))*self.hop, self.hop)
        N = int(self.fs/float(self.fr))
        padded_window_size = len(h)
        f = self.fs*np.linspace(0, 0.5, int(np.round(N/2)), endpoint=True)
        Lh = int(np.floor(float(padded_window_size-1) / 2))
        tfr = np.zeros((int(N), len(t)), dtype=np.float)     
            
        for icol in range(0, len(t)):
            ti = int(t[icol])           
            tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                            int(min([round(N/2.0)-1, Lh, len(x)-ti])))
            indices = np.mod(N + tau, N) + 1                                             
            tfr[indices-1, icol] = x[ti+tau-1] * h[Lh+tau-1] #\
    #                                /np.linalg.norm(h[Lh+tau-1])           
        tfr = scipy.fftpack.fft(tfr, n=N, axis=0)
        return tfr, f, t, N
    
    def STFT_magnitude(self, x, h):
        tfr, f, t, N = self.STFT_complex(x, h)                                
        tfr = abs(tfr)  
        return tfr, f, t, N

    def dwindow(self, h):
        # h = self.get_window(self) 
        h = np.reshape(h, (-1, 1))
        hrow,hcol=np.array(h).shape
        if hcol!=1:
            print('h must have only one column')

        Lh=(hrow-1)/2.0
        step_height=(h[0]+h[hrow-1])/2
        ramp=(h[hrow-1]-h[0])/(hrow-1)
        L = np.arange(-Lh,Lh+1,1, dtype=float)
        L_T = L.reshape(-1,1)
        h2 = h-step_height-ramp*L_T
        h2 = np.append([0], h2)
        h2 = np.append(h2, [0])
        Dh = (h2[2:hrow+2]-h2[0:hrow])/2 + ramp
    #    Dh = np.reshape(Dh, (-1, 1))
        return Dh

    def phase_deri(self, x):
        h = self.get_window() 
        Lh = (len(h)-1)/2
        Dh = self.dwindow(h)
        Th = h*np.arange(-Lh,Lh+1) 
        DDh = self.dwindow(Dh)
        TDh = Dh*np.arange(-Lh,Lh+1) 
        tfr, f, t, N = self.STFT_complex(x, h)
        tfrD, f, t, N = self.STFT_complex(x, Dh)
        tfrT, f, t, N = self.STFT_complex(x, Th)
        tfrDD, f, t, N = self.STFT_complex(x, DDh)
        tfrTD, f, t, N = self.STFT_complex(x, TDh)
        
        tfr[abs(tfr)<1E-6] = 1E10
        ifd = np.imag(tfrD/tfr)*(N/2/np.pi)
        gdf = np.real(tfrT/tfr/self.hop) # actually -gdf
        fm = np.imag(N*(tfrD*tfrD-tfrDD*tfr)/(tfrT*tfrD-tfrTD*tfr)/2/np.pi)
        ifd = ifd[:int(round(N/2)),:]
        gdf = gdf[:int(round(N/2)),:]
        tfr = tfr[:int(round(N/2)),:]
        return ifd, gdf, tfr, fm

    def inst_freq(self, x, locs, central_freq, N):
        IFD, GDF, tfr, FM = self.phase_deri(x)
        ifff = np.zeros((1, locs.shape[1]))
        for ii in range(locs.shape[1]): #range(IFD.shape[1]): #
            index_low = int(np.floor(central_freq[int(locs[0,ii])-3]/(self.fs/N)))
            index_high = int(np.ceil(central_freq[int(locs[0,ii])+3]/(self.fs/N)))
            if index_low > 0 and index_high-index_low > 1:
                temp = abs(tfr[index_low:index_high, ii])
                k = np.argmax(temp)
                k_max = index_low+k
                if IFD[k_max, ii] < index_high-index_low:
                    Deviation = IFD[k_max, ii] + FM[k_max, ii]*GDF[k_max, ii]
                    ifff[0, ii] = (index_low+k-Deviation-1)*(self.fs/N)
        return ifff

    def Freq2LogFreqMapping(self, tfr, f):
        StartFreq = self.fc
        StopFreq = 1/self.tc
        Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*self.NumPerOct)
        central_freq = []

        for i in range(-self.bov, Nest+self.bov):
            CenFreq = StartFreq*pow(2, float(i)/self.NumPerOct)
            central_freq.append(CenFreq)

        Nest = len(central_freq)
        freq_band_transformation = np.zeros((Nest, len(f)), dtype=np.float)
        for i in range(self.bov, Nest-self.bov):
            l = int(round(central_freq[i-self.bov]/self.fr))
            r = int(round(central_freq[i+self.bov]/self.fr)+1)
            BW = central_freq[i+self.bov] - central_freq[i-self.bov]
            if l >= r-1:
                freq_band_transformation[i, l] = 1
            else:
                for j in range(l, r):
                    if f[j] > central_freq[i-self.bov] and f[j] < central_freq[i]:
                        freq_band_transformation[i, j] = (f[j] - central_freq[i-self.bov]) / (central_freq[i] - central_freq[i-self.bov]) / BW
                    elif f[j] > central_freq[i] and f[j] < central_freq[i + self.bov]:
                        freq_band_transformation[i, j] = (central_freq[i + self.bov] - f[j]) / (central_freq[i + self.bov] - central_freq[i]) / BW
        tfrL = np.dot(freq_band_transformation, tfr)
        tfrL = tfrL[self.bov:-self.bov,:]
        central_freq = central_freq[self.bov:-self.bov]
        return tfrL, central_freq

    # construct a quefrency-domain triangular filterbank
    def Quef2LogFreqMapping(self, ceps, q):
        StartFreq = self.fc
        StopFreq = 1/self.tc
        Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*self.NumPerOct)
        central_freq = []

        for i in range(-self.bov, Nest+self.bov):
            CenFreq = StartFreq*pow(2, float(i)/self.NumPerOct)
            central_freq.append(CenFreq)
            
        f = 1/(q+1E-10) # avoid divide by zero
        Nest = len(central_freq)
        freq_band_transformation = np.zeros((Nest, len(f)), dtype=np.float)
        for i in range(self.bov, Nest-self.bov):
            BW = central_freq[i+self.bov] - central_freq[i-self.bov]
            for j in range(int(round(self.fs/central_freq[i+self.bov])), int(round(self.fs/central_freq[i-self.bov])+1)):
                if f[j] > central_freq[i-self.bov] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i-self.bov])/(central_freq[i] - central_freq[i-self.bov]) / np.sqrt(BW)
                elif f[j] > central_freq[i] and f[j] < central_freq[i+self.bov]:
                    freq_band_transformation[i, j] = (central_freq[i + self.bov] - f[j]) / (central_freq[i + self.bov] - central_freq[i]) / np.sqrt(BW)

        tfrL = np.dot(freq_band_transformation, ceps)
        tfrL = tfrL[self.bov:-self.bov,:]
        central_freq = central_freq[self.bov:-self.bov]
        return tfrL, central_freq

    def CFP_filterbank(self, x):
        NumofLayer = np.size(self.g)
        print(self.win)
        h = self.get_window()
        tfr, f, t, N = self.STFT_magnitude(x, h)
        tfr = np.power(abs(tfr), self.g[0])
        tfr0 = tfr # original STFT
        ceps = np.zeros(tfr.shape)

        if NumofLayer >= 2:
            for gc in range(1, NumofLayer):
                if np.remainder(gc, 2) == 1:
                    tc_idx = round(self.fs*self.tc)
                    ceps = np.real(np.fft.fft(tfr, axis=0))/np.sqrt(N)
                    ceps = utils.nonlinear_func(ceps, self.g[gc], tc_idx)
                else:
                    fc_idx = round(self.fc/self.fr)
                    tfr = np.real(np.fft.fft(ceps, axis=0))/np.sqrt(N)
                    tfr = utils.nonlinear_func(tfr, self.g[gc], fc_idx)

        tfr0 = tfr0[:int(round(N/2)),:] # original STFT
        tfr = tfr[:int(round(N/2)),:] # the last frequency-domain feature
        ceps = ceps[:int(round(N/2)),:] # the last quefrency-domain feature

        q = np.arange(int(round(N/2)))/float(self.fs)
        
        tfrL0, central_frequencies = self.Freq2LogFreqMapping(tfr0, f)
        tfrLF, central_frequencies = self.Freq2LogFreqMapping(tfr, f)
        tfrLQ, central_frequencies = self.Quef2LogFreqMapping(ceps, q)

        return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies 

class Note_level_processing():    
    def __init__(self, SF_maxfilt_bw, SF_thres, lam, Note_th, Repeated_note_th, Transition_th):
        # super().__init__()
        # self.CFP_HMLC_module = CFP_HMLC_module
        self.SF_maxfilt_bw = SF_maxfilt_bw
        self.SF_thres = SF_thres
        self.lam = lam
        self.Note_th = Note_th 
        self.Repeated_note_th = Repeated_note_th 
        self.Transition_th = Transition_th
        
    def spectral_flux(self, tfrL0):
        tfrL0_max = scipy.ndimage.maximum_filter1d(tfrL0, size = self.SF_maxfilt_bw, axis = 0)
        tfrL0_m = scipy.ndimage.maximum_filter1d(tfrL0, size = 3, axis = 1)
        tfrL0_max2 = np.maximum(np.roll(tfrL0_m, 3, axis=0), np.roll(tfrL0_m, -3, axis=0))
        SF = np.zeros(tfrL0.shape[1])
        for i in range(1, tfrL0.shape[1]-1):
            D = tfrL0[:,i+1] - np.maximum(tfrL0_max[:,i-1], tfrL0_max2[:,i-1]) #tfrL0_max[:,i-1] 
            D[D<0.0]=0.0
            D[D>0.0]=1.0
            SF[i] = np.sum((np.abs(D)+D)/2)    
        # SF = np.convolve(SF, scipy.signal.hanning(21)/21.0, mode='same')
    #    SF = np.convolve(SF, np.ones(21)/21.0, mode='same')
        _, onset = utils.findpeaks_time(SF, self.SF_thres)
        return SF, onset

    def note_to_note_value(self, beats, Note):
        # triplet -> 3 notes in 2n beats and they are in equal length (max/min<1.1)
        # determine sub-beat notes (1/4 beats)
        # one sub-beat for only one note
        onset_events = []
        duration = []
        for i in range(len(Note)):
            onset_events.append(Note[i]['time'][0,0])
            duration.append(Note[i]['time'].shape[1])
        onset_events = np.asarray(onset_events)
        duration = np.asarray(duration)
        
        # detect on-beat notes -> how many notes in how many beats (10% of period?)
        metric_position = np.zeros(onset_events.shape[0])    
        note_value = np.zeros(onset_events.shape[0]) 
        beats = np.asarray(beats)
               
        for i in range(beats.shape[0]):
            beat_period = beats[min(i+1,beats.shape[0]-1)]-beats[i]
            for j in range(onset_events.shape[0]):
                if onset_events[j]>=beats[i] and onset_events[j]<beats[min(len(beats)-1,i+1)]:                  
                    ratio_n = (onset_events[j]-beats[i])/float(beat_period)
                    metric_position[j] = i+0.25*round(ratio_n/0.25)  
        
        for i in range(beats.shape[0]):
            beat_period = beats[min(i+1,beats.shape[0]-1)]-beats[i]
            for j in range(onset_events.shape[0]):
                if onset_events[j]>=beats[i] and onset_events[j]<beats[min(len(beats)-1,i+1)]: 
                    ratio_n = duration[j]/float(beat_period)
                    note_value[j]=0.25*round(ratio_n/0.25)                    
        return metric_position, note_value
        
    def note_seg_dp(self, Z, onset_time):
        Note_locs = np.zeros((2, Z.shape[1]))
        # lam = 1E-3
        for i in range(len(onset_time)-1):
            Z_ = Z[:, np.arange(onset_time[i], onset_time[i+1]+1)]
            Z_ = Z_-np.tile(np.reshape(np.mean(Z_, axis=1), (-1,1)), (1, Z_.shape[1]))
            Z_[Z_<0]=0
            _, contour, contour_val = utils.DPNT(Z_, self.lam)
            locs = np.vstack((contour, contour_val))        
            Note_locs[:, onset_time[i]:onset_time[i+1]+1] = locs #np.hstack((Note_locs, locs))      
        return Note_locs

    def post_proc(self, contour, contour_fine, onset_SF): # segment the pitch-based onset
        # resolution: 20 cents
        pitch_diff_th = self.Transition_th
        time_diff_th = self.Note_th
    #    contour = 60*np.log2((ifff[0,:]+1E-8)/CenFreq[0])

    ############# onset segmentation ##############
        contour_diff = np.abs(np.diff(contour)) # coarse
        onset = np.where(contour_diff>=pitch_diff_th)[0]+1
        onset = np.append(onset, np.reshape(onset_SF,[-1,1]))
        onset = np.sort(onset)
        
        contour_fine_diff = np.abs(np.diff(contour_fine)) # fine
        onset_fine = np.where(contour_fine_diff>=pitch_diff_th)[0]+1
        onset_fine = np.append(onset_fine, np.reshape(onset_SF,[-1,1]))
        onset_fine = np.sort(onset_fine)

    ############ remove short onsets ###############    
        R_idx = []
        for i in range(len(onset)-1):
            if onset[i+1]-onset[i] < time_diff_th:
                R_idx.append(i)
        for index in sorted(R_idx, reverse=True):
            onset = np.delete(onset,index)
            
        R_idx = []
        for i in range(len(onset_fine)-1):
            if onset_fine[i+1]-onset_fine[i] < time_diff_th:
                R_idx.append(i)
        for index in sorted(R_idx, reverse=True):
            onset_fine = np.delete(onset_fine,index)
            
    ############ merge short onsets with repeated notes ###############    
        R_idx = []
        for i in range(len(onset)-2):
            pitch_now = np.median(contour[onset[i]:onset[i+1]]) #contour[onset[i+1]-1] #
            pitch_next = np.median(contour[onset[i+1]:onset[i+2]]) #contour[onset[i+1]] #
            duration_now = onset[i+1]-onset[i]
            duration_next = onset[i+2]-onset[i+1] 
            if abs(pitch_now-pitch_next)<5 and min(duration_now, duration_next) <= self.Repeated_note_th:
                R_idx.append(i+1)
        for index in sorted(R_idx, reverse=True):
            onset = np.delete(onset,index)
            
        R_idx = []
        for i in range(len(onset_fine)-2):
            pitch_now = np.median(contour_fine[onset_fine[i]:onset_fine[i+1]]) #contour_fine[onset_fine[i+1]-1] #
            pitch_next = np.median(contour_fine[onset_fine[i+1]:onset_fine[i+2]]) #contour_fine[onset_fine[i+1]] #
            duration_now = onset_fine[i+1]-onset_fine[i]
            duration_next = onset_fine[i+2]-onset_fine[i+1] 
            if abs(pitch_now-pitch_next)<5 and min(duration_now, duration_next) <= self.Repeated_note_th:
                R_idx.append(i+1)
        for index in sorted(R_idx, reverse=True):
            onset_fine = np.delete(onset_fine,index)

    ############ note list generation ##############    
        Note_coarse = []
        for i in range(len(onset)-1):
            offset = onset[i+1]-1
    #        if np.where(np.abs(np.diff(contour[onset[i]:offset+1]))>=pitch_diff_th)[0].shape[0]>0:
    ##            print(np.where(np.abs(np.diff(contour[onset[i]:offset+1]))>=pitch_diff_th)[0].shape[0])
    #            offset = onset[i]+np.where(np.abs(np.diff(contour[onset[i]:offset+1]))>=pitch_diff_th)[0][0]
    #        print(offset)
            midi_pitch = np.median(contour[onset[i]:offset+1])*np.ones((1,offset-onset[i]+1))
            No = {'time': np.reshape(np.arange(onset[i],offset+1), [1,-1]),\
                  'contour': np.reshape(contour[onset[i]:offset+1], [1,-1]),\
                  'pitch': midi_pitch}          
            Note_coarse.append(No)
            
        Note_fine = []
        for i in range(len(onset_fine)-1):
            offset = onset_fine[i+1]-1
    #        if np.where(np.abs(np.diff(contour_fine[onset_fine[i]:offset+1]))>=pitch_diff_th)[0].shape[0]>0:
    #            offset = onset_fine[i]+np.where(np.abs(np.diff(contour_fine[onset_fine[i]:offset+1]))>=pitch_diff_th)[0][0]
            midi_pitch_fine = np.median(contour_fine[onset_fine[i]:offset+1])*np.ones((1,offset-onset_fine[i]+1))
            No = {'time': np.reshape(np.arange(onset_fine[i],offset+1), [1,-1]),\
                  'contour': np.reshape(contour_fine[onset_fine[i]:offset+1], [1,-1]),\
                  'pitch': midi_pitch_fine}          
            Note_fine.append(No)
            
        Note = {'coarse': Note_coarse, 'fine': Note_fine}
        return Note

class CFP_HMLC_tempo_beat():
    def __init__(self, CFP_HMLC_module):
        self.CFP_HMLC_module = CFP_HMLC_module
        
    def estimate_tempo_beat(self, SF):
        
        tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = self.CFP_HMLC_module.CFP_filterbank(SF)
        
        # CFP-HMLC tempogram computation
        har_num = [60, 95, 120, 140, 155]
        Z = tfrLF * tfrLQ
        if self.CFP_HMLC_module.Har > 1:
            for i in range(self.CFP_HMLC_module.Har-1):
                Z[:-har_num[i],:] = Z[:-har_num[i],:]*tfrLF[har_num[i]:,:]
        
        # complex-valued Fourier tempogram for PLP extraction
        h = self.CFP_HMLC_module.get_window()
        SF_spec, f, t, N = self.CFP_HMLC_module.STFT_complex(SF, h) 
        
        # PLP extraction
        mask = np.zeros(SF_spec.shape)
        _, contour, _ = utils.DPNT(Z, 1E0)
        
        tempo = []
        for i in range(Z.shape[1]):
            print(contour[i])
            tempo.append(CenFreq[int(contour[i])]) #FS/P #1.78 #
    #    print('Estimated tempo: ', 60*tempo, ' BPM')
            for j in range(1):
                mask[int(0.8*(j+1)*tempo[i]*N/self.CFP_HMLC_module.fs):int(1.2*(j+1)*tempo[i]*N/self.CFP_HMLC_module.fs),:]=1 
                mask[int(-1.2*(j+1)*tempo[i]*N/self.CFP_HMLC_module.fs):int(-0.8*(j+1)*tempo[i]*N/self.CFP_HMLC_module.fs),:]=1 
        SF_spec = np.multiply(SF_spec, mask)
            
        Lh = int(np.floor(float(self.CFP_HMLC_module.win-1) / 2)) 
        PLP = np.zeros(SF.shape)
        for i in range(SF_spec.shape[1]):
            ti = int(t[i])  
            tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                            int(min([round(N/2.0)-1, Lh, len(SF)-ti])))
            SF_recon = np.real(scipy.fftpack.ifft(SF_spec[:,i]))
            SF_recon = scipy.fftpack.fftshift(SF_recon)
            PLP[ti+tau-1] = PLP[ti+tau-1] + SF_recon[Lh+tau]#*h[Lh+tau-1]
            
        tempo = np.asarray(tempo)
        _, beats = utils.findpeaks_time(PLP, 0)
        return tempo, beats, PLP, Z, CenFreq

class note_transcription(CFP_HMLC, Note_level_processing):    #CFP_HMLC, Note_level_processing
    def __init__(self, CFP_HMLC_module, Note_level_module):
        self.CFP_HMLC_module = CFP_HMLC_module
        self.Note_level_module = Note_level_module
        
    def feature_extraction(self, x): #filename
        print(x.shape)
        x = x.astype('float32')
        
        # MLC computation
        tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = self.CFP_HMLC_module.CFP_filterbank(x)
        
        # CFP-HMLC computation
        har_num = [60, 95, 120, 140, 155]
        Z = tfrLF * tfrLQ
        if self.CFP_HMLC_module.Har > 1:
            for i in range(self.CFP_HMLC_module.Har-1):
                Z[:-har_num[i],:] = Z[:-har_num[i],:]*tfrLF[har_num[i]:,:]

        # Optional, just to fit sitar's pitch range
        Z[135:,:] = 0
        Z[:5,:] = 0
        
        # Optional, this part is to avoid great leaps in the transcription results
        Z_test = scipy.signal.convolve(Z, np.ones((70, 200)), mode='same')
        for i in range(Z.shape[1]):
            tt = np.argmax(Z_test[:,i])
            Z[:tt-35,i] = 0
            Z[tt+35:,i] = 0
        
        # Normalize the frames
        Z = Z/np.tile(np.reshape(np.sum(Z, axis=0), (1,-1))+1E-8, (Z.shape[0], 1))
        Z[Z<0.0]=0.0
        
        # Onset detection
        SF, onset_time = self.Note_level_module.spectral_flux(tfrL0)
        onset_time = onset_time[0]

        # Pitch contour extraction from segments
        note_locs = self.Note_level_module.note_seg_dp(Z[:136,:], onset_time) 

        # Pitch contour refinement with 2nd-order SST
        # IFD, GDF, tfr_c, FM = self.phase_deri(self, x, h)
        N = int(self.CFP_HMLC_module.fs/float(self.CFP_HMLC_module.fr))
        ifff = self.CFP_HMLC_module.inst_freq(x, note_locs, CenFreq, N) #(note_locs, IFD, GDF, FM, tfr_c, CenFreq, fs, N)

        return Z, tfrL0, note_locs, CenFreq, ifff, SF, onset_time
    
    def transcribe_note(self, filename):
        x, fs = sf.read(filename)
        if len(x.shape)>1:
            x = np.mean(x, axis = 1)
        x = scipy.signal.resample_poly(x, 16000, fs)
        fs = 16000
        T = int(np.floor(len(x)/fs/10.0))
        i = 0

        Note_all = {'coarse': [], 'fine': []} #[]
        
        # To save memory, divide the audio clips into 10-sec segments
        while i<=T:
            xx = x[10*i*fs:np.min((10*i*fs+11*fs, len(x)))] 
            if xx.shape[0] > 100:
                Z, S, locs, CenFreq, ifff, SF, onset_time = self.feature_extraction(xx)
                contour = 60*np.log2((ifff[0,:]+1E-8)/CenFreq[0])    
                Note = self.Note_level_module.post_proc(locs[0,:], contour, onset_time)

                for j in range(len(Note['coarse'])):
                    if Note['coarse'][j]['time'][0,0]>1000:
                        Note['coarse'] = Note['coarse'][0:j] 
                        break
                    else:
                        Note['coarse'][j]['time'] = Note['coarse'][j]['time'] + 1000*i
                        
                for j in range(len(Note['fine'])):
                    if Note['fine'][j]['time'][0,0]>1000:
                        Note['fine'] = Note['fine'][0:j] 
                        break
                    else:
                        Note['fine'][j]['time'] = Note['fine'][j]['time'] + 1000*i

                print(len(Note['coarse']))
                print(len(Note['fine']))
                Note_all['coarse'].extend(Note['coarse'])
                Note_all['fine'].extend(Note['fine'])
    #            Note_fine_all.extend(Note_fine)
                if i == 0:
                    Z_all = Z[:,:min(1000, Z.shape[1])]
                    S_all = S[:,:min(1000, S.shape[1])]
                    SF_all = SF[:min(1000, SF.shape[0])]
                else:
                    Z_all = np.append(Z_all, Z[:,:min(1000, Z.shape[1])], axis = 1)
                    S_all = np.append(S_all, S[:,:min(1000, S.shape[1])], axis = 1)
                    SF_all = np.append(SF_all, SF[:min(1000, SF.shape[0])])
            i = i+1
        Note_all['coarse'] = utils.remove_outlier(Note_all['coarse'])
        Note_all['fine'] = utils.remove_outlier(Note_all['fine'])
        return Note_all, Z_all, S_all, SF_all, CenFreq