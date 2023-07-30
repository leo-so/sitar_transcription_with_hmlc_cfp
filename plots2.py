# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 23:18:00 2022

@author: lisow
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class plots2:
    def __init__(self):
        self.axis_font = {'fontname':'Arial', 'size':'14'}  
        self.raga_list = np.array([0, 9, 24, 34, 49, 59, 69, 84, 94, 109, 119, 129])
        self.boundary_sec = np.array([0, 19, 38, 57, 75, 92, 106, 124, 142, 159, 177, 194, 212, 230, 247])
        self.rep_sec = np.array([[35,38], [52,57], [72,75], [89,93], [103,106], [119,124], [151,154], [173,178], [192,196], [208,213], [244,248]])
        self.mext_sec = np.array([1, 2, 3, 20, 28, 45.5, 88, 89, 114.5, 184.5, 235.8]) #

    def plot_contour(self, Note, figs, ax):
        plot_color = np.zeros((len(Note), 3))
        for i in range(len(Note)):
            plot_color[i,:] = 0.9*np.random.rand(1, 3)
        ax.set_ylabel('p (12-ET)', **self.axis_font)
        ax.set_yticks((0, 59, 119))
        ylabels = ('C3', 'C4', 'C5')
        ax.set_yticklabels(ylabels, **self.axis_font)
        NumOfSegments = len(Note) #int(np.amax(labels))
        for j in range(NumOfSegments):
    #        ax.plot(Note[j]['time'][0,:], Note[j]['pitch'][0,:], linestyle='-', c=[0.0,0.0,0.0], linewidth=2)
            ax.plot(Note[j]['time'][0,:], Note[j]['contour'][0,:], linestyle='-', c=plot_color[j,:], linewidth=2)        
            ax.plot(Note[j]['time'][0,0], Note[j]['contour'][0,0], marker='o', c=plot_color[j,:], markersize=3) #onset_pitch[j]
        
    #    ax.set_xlim([0,2000])
        ax.set_ylim([0, 135]) 
    

    def plot_fig(self, Note, figs, ax):  #filename):
        # raga_list = np.array([0, 9, 24, 34, 49, 59, 69, 84, 94, 109, 119, 129])
        ax.set_ylabel('p (12-ET)', **self.axis_font)
        
    #    ax.set_yticks((0, 9, 24, 34, 49, 59, 69, 84, 94, 109, 119, 129))
    #    ylabels = ('C3', 'D3', 'F3', 'G3', 'Bb3', 'C4', 'D4', 'F4', 'G4', 'Bb4', 'C5', 'D5')
        # simplified version
        ax.set_yticks((0, 59, 119))
        ylabels = ('C3', 'C4', 'C5')
        ax.set_yticklabels(ylabels, **self.axis_font)
        
    #    plt.imshow(Z, origin= 'lower', aspect = 'auto', cmap = 'Greys')
    #    plt.show()
        ax.hlines(y=0, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # C3    
        ax.hlines(y=9, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # D3
        ax.hlines(y=24, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # F3
        ax.hlines(y=34, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # G3
        ax.hlines(y=49, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # Bb3
        ax.hlines(y=59, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # C4
        ax.hlines(y=69, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # D4
        ax.hlines(y=84, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # F4
        ax.hlines(y=94, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # G4
        ax.hlines(y=109, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # Bb4
        ax.hlines(y=119, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # C5
        ax.hlines(y=129, xmin=0, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) # D5
        
        NumOfSegments = len(Note) #int(np.amax(labels))
        for j in range(NumOfSegments):
            note_ = self.raga_list[np.where(abs(Note[j]['pitch'][0,0]-self.raga_list)== \
                            np.amin(abs(Note[j]['pitch'][0,0]-self.raga_list)))[0][0]]
            ax.plot(Note[j]['time'][0,:], note_*np.ones(Note[j]['pitch'].shape[1]), linestyle='-', c=[0.0,0.0,0.0], linewidth=2)
    #        ax.plot(Note[j]['time'][0,:], Note[j]['contour'][0,:], linestyle='-', c=plot_color[j,:], linewidth=2)        
    #        ax.plot(Note[j]['time'][0,0], Note[j]['contour'][0,0], marker='x', c=plot_color[j,:]) #onset_pitch[j]
            ax.plot(Note[j]['time'][0,0], note_, marker='o', c=[0.0,0.0,0.0], markersize=3)
        
        ax.set_ylim([0, 135]) 
    

    def plot_note_value(self, Note, note_value, figs, ax):  #filename):
        ax.set_ylabel('v (Matra)', **self.axis_font)
        
    #    ax.set_yticks((0, 9, 24, 34, 49, 59, 69, 84, 94, 109, 119, 129))
    #    ylabels = ('C3', 'D3', 'F3', 'G3', 'Bb3', 'C4', 'D4', 'F4', 'G4', 'Bb4', 'C5', 'D5')
        # simplified version
        ax.set_yticks((0, 1, 2, 3))
        ylabels = ('1/8', '1/4', '1/2', '1')
        ax.set_yticklabels(ylabels, **self.axis_font)
        
        ax.hlines(y=0, xmin=-1, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) 
        ax.hlines(y=1, xmin=-1, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) 
        ax.hlines(y=2, xmin=-1, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) 
        ax.hlines(y=3, xmin=-1, xmax=2000, color =(0.9, 0.9, 0.9), linewidth=2) 
        NumOfSegments = len(Note) #int(np.amax(labels))
        for j in range(NumOfSegments):
    #        note_ = raga_list[np.where(abs(Note[j]['pitch'][0,0]-raga_list)== \
    #                        np.amin(abs(Note[j]['pitch'][0,0]-raga_list)))[0][0]]
            note_value_ = np.log2(4*note_value+1E-10)
    #        ax.plot(Note[j]['time'][0,:], note_*np.ones(Note[j]['pitch'].shape[1]), linestyle='-', c=[0.0,0.0,0.0], linewidth=2)
    #        ax.plot(Note[j]['time'][0,:], Note[j]['contour'][0,:], linestyle='-', c=plot_color[j,:], linewidth=2)        
    #        ax.plot(Note[j]['time'][0,0], Note[j]['contour'][0,0], marker='x', c=plot_color[j,:]) #onset_pitch[j]
            ax.plot(Note[j]['time'][0,0], note_value_[j], marker='x', c=[0.0,0.0,0.0], markersize=5)    
        ax.set_ylim([-1, 4]) 
    
    def plot_fig2(self, S, Z, SF, TP, tempo, ffff, PLP, beats, Note_all, note_value, savename):    
        figs, ax = plt.subplots(8, 2, gridspec_kw={'width_ratios': [1, 0.02], 
                            'wspace': 0.04, 'hspace':0.1,                
                            'height_ratios': [1, 1, 0.5, 1, 0.5, 1, 1, 1]}, figsize=(8, 10)) # figsize=(6.7, 8)
                
        pos = ax[0,0].imshow(S, origin='lower', interpolation='bilinear', aspect='auto')
        ax[0,0].set_xticks([])
        cb = figs.colorbar(pos, cax=ax[0,1])
        ax[0,1].set_yticklabels(ax[0,1].get_yticklabels(), **self.axis_font)
        ax[0,0].set_yticks([37, 116, 176, 236, 296])
        ax[0,0].set_yticklabels([0.2, 0.5, 1, 2, 4], **self.axis_font)
        ax[0,0].set_ylabel('f (kHz)', **self.axis_font)
        
        pos = ax[1,0].imshow(Z, origin='lower', interpolation='bilinear', aspect='auto')
        ax[1,0].set_xticks([])
        #cb = figs.colorbar(pos, cax=ax[1,1])
        #ax[1,1].set_yticklabels(ax[1,1].get_yticklabels(), **axis_font)
        ax[1,0].set_yticks([37, 116, 176, 236, 296])
        ax[1,0].set_yticklabels([0.2, 0.5, 1, 2, 4], **self.axis_font)
        ax[1,0].set_ylabel('f (kHz)', **self.axis_font)
        #        plot_fig(Note_all['coarse'], figs, ax[0,0])
        #        ax[0,0].set_xlim(0,len(Note_all['coarse'])-1)
        
        ax[2,0].plot(SF)
        ax[2,0].set_xlim(0, SF.shape[0])
        ax[2,0].set_yticklabels((0,0,50), **self.axis_font)
        ax[2,0].set_yticks([])   
        ax[2,0].set_xticks([])
            
        
        pos = ax[3,0].imshow(TP, origin='lower', interpolation='bilinear', aspect='auto')
        ax[3,0].set_xticks([])
        #cb = figs.colorbar(pos, cax=ax[3,1])
        #ax[3,1].set_yticklabels(ax[3,1].get_yticklabels(), **axis_font)
        ax[3,0].set_yticks([31, 91, 151])
        ax[3,0].set_yticklabels([60, 120, 180], **self.axis_font)
        ax[3,0].set_ylabel('t (BPM)', **self.axis_font)
        
        tempo_idx = np.zeros(tempo.shape)
        for i in range(tempo.shape[0]):
            tempo_idx[i] = np.where(np.abs(tempo[i]-ffff)==np.amin(np.abs(tempo[i]-ffff)))[0]
        ax[3,0].plot(tempo_idx, 'r--')
        
        ax[4,0].plot(PLP)
        ax[4,0].set_xlim(0, SF.shape[0])
        ax[4,0].set_yticks([]) 
        ax[4,0].set_xticks([])
        
        for i in range(beats[0].shape[0]):
            ax[5,0].vlines(beats[0][i], 0, 135, color = [0.9, 0.9, 0.9])
            ax[6,0].vlines(beats[0][i], 0, 135, color = [0.9, 0.9, 0.9])
            ax[7,0].vlines(beats[0][i], -1, 135, color = [0.9, 0.9, 0.9])
        
        self.plot_contour(Note_all['coarse'], figs, ax[5,0])
        ax[5,0].set_xlim(0, SF.shape[0])
        ax[5,0].set_xticks([])
            
        self.plot_fig(Note_all['coarse'], figs, ax[6,0])
        ax[6,0].set_xlim(0, SF.shape[0])
        ax[6,0].set_xticks([])
        
        self.plot_note_value(Note_all['coarse'], note_value, figs, ax[7,0])
        ax[7,0].set_xlim(0, SF.shape[0])
        ax[7,0].set_xticks((0, 500, 1000, 1500))
        xlabels = (0, 5, 10, 15)
        ax[7,0].set_xticklabels(xlabels, **self.axis_font)
        ax[7,0].set_xlabel('Time (s)', **self.axis_font)
        
        ax[0,1].axis('off')
        ax[1,1].axis('off')
        ax[2,1].axis('off')
        ax[3,1].axis('off')
        ax[4,1].axis('off')
        ax[5,1].axis('off')
        ax[6,1].axis('off')
        ax[7,1].axis('off')
        
        figs.savefig(savename + '.png', dpi=300, frameon='false')
        
    def plot_fig6(self, tempo, savename):
        plt.figure(figsize=(10,4))
        for i in range(self.boundary_sec.shape[0]):
            plt.vlines(self.boundary_sec[i], -0.1, 180.1, color = (0.5, 0.5, 0.5))
        plt.plot(np.arange(-2, tempo.shape[0]+2)/5.0, np.convolve(tempo*60, np.ones(5)/5.0))
        plt.xlim(0,250)
        plt.ylim(60,150)
        plt.xticks([1, 50, 100, 150, 200, 250], **self.axis_font)
        plt.yticks([60, 120], **self.axis_font)
        plt.xlabel('Time (sec)', **self.axis_font)
        plt.ylabel('Tempo (BPM)', **self.axis_font)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        plt.savefig(savename + '.png', dpi=300, frameon='false')
        
        
    def plot_fig7(self, SSM, struct_bound_, rep_, ext_time, savename):
        ######## plot the SSM and structure analysis results!
        
        # boundary_sec = np.array([0, 19, 38, 57, 75, 92, 106, 124, 142, 159, 177, 194, 212, 230, 247])
        # rep_sec = np.array([[35,38], [52,57], [72,75], [89,93], [103,106], [119,124], [151,154], [173,178], [192,196], [208,213], [244,248]])
        # mext_sec = np.array([1, 2, 3, 20, 28, 45.5, 88, 89, 114.5, 184.5, 235.8]) #

        boundary_frame = self.boundary_sec*5 #sr/float(hop_length)
        rep_frame = self.rep_sec*5
        mext_frame = self.mext_sec*5

        figs, ax = plt.subplots(4, 2, gridspec_kw={'width_ratios': [1, 0.05], 
                                  'wspace': 0.05, 'hspace':0.1,                
                                  'height_ratios': [1, 0.1, 0.1, 0.1]}, figsize=(6, 8)) # figsize=(6.7, 8)
        
        pos = ax[0,0].imshow(SSM, origin='lower', interpolation='bilinear', aspect='auto')
        ax[0,0].set_xticks([])
        ax[0,0].set_yticks([])
        cb = figs.colorbar(pos, cax=ax[0,1])
        ax[0,1].set_yticklabels((0,1,2,3,4,5,6,7), **self.axis_font)
        
        for i in range(boundary_frame.shape[0]):
            ax[1,0].vlines(boundary_frame[i], -0.1, 1.1, color = (0.5, 0.5, 0.5))
        
        for i in range(rep_frame.shape[0]):
            ax[2,0].add_patch(patches.Rectangle((rep_frame[i,0], 0), rep_frame[i,1]-rep_frame[i,0], 1,\
                              edgecolor = (0.93, 0.80, 0.93), facecolor = (0.93, 0.80, 0.93), fill=True))
        
        for i in range(mext_frame.shape[0]):
            ax[3,0].vlines(mext_frame[i], -0.1, 1.1, color = (0.5, 0.5, 0.5), linestyle=':')
        
        yticks = (0, 300, 600, 900, 1200)
        ax[0,0].set_yticks(yticks)
        ylabels = (0, 60, 120, 180, 240)
        ax[0,0].set_yticklabels(ylabels, **self.axis_font)
        ax[0,0].set_xticks([])
        ax[0,0].set_ylabel('Time (s)', **self.axis_font)
        
        ax[1,0].plot(struct_bound_, color = (0.11, 0.14, 0.59))
        ax[1,0].set_xlim(0,1250)
        ax[1,0].set_ylim(0,1)
        ax[1,0].set_xticks([])
        ax[1,0].set_yticks([])
        
        ax[2,0].plot(rep_, color = (0.11, 0.14, 0.59))
        ax[2,0].set_xlim(0,1250)
        ax[2,0].set_ylim(0,1)
        ax[2,0].set_xticks([])
        ax[2,0].set_yticks([])
        
        ax[3,0].plot(ext_time, 0.5*np.ones(ext_time.shape[0]), marker='x', linestyle='', color = (0.11, 0.14, 0.59))
        
        xticks = (0, 300, 600, 900, 1200)
        ax[3,0].set_xticks(xticks)
        xlabels = (0, 60, 120, 180, 240)
        ax[3,0].set_xticklabels(xlabels, **self.axis_font)
        ax[3,0].set_xlabel('Time (s)', **self.axis_font)
        ax[3,1].axis('off')
        ax[2,1].axis('off')
        ax[1,1].axis('off')
        
        figs.savefig(savename + '.png', dpi=200, frameon='false')