# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:29:45 2020

@author: lisu
"""
import numpy as np

def global_align(x, y, gap, match, mismatch):
   
    A = []
    for i in range(len(y) + 1):
        A.append([0] * (len(x) +1))
    for i in range(len(y)+1):
        A[i][0] = gap * i
    for i in range(len(x)+1):
        A[0][i] = gap * i
    for i in range(1, len(y) + 1):
        for j in range(1, len(x) + 1):
           
            A[i][j] = max(
            A[i][j-1] + gap,
            A[i-1][j] + gap,
            A[i-1][j-1] + (match if y[i-1] == x[j-1] else mismatch)
            )

    align_X = []
    align_Y = []
    opt_loc = []
    i = len(x)
    j = len(y)

    while i > 0 or j > 0:
         
        current_score = A[j][i]

        if i > 0 and j > 0 and x[i - 1] == y[j - 1]:
            align_X = [x[i - 1]] + align_X
            align_Y = [y[j - 1]] + align_Y
            opt_loc.append([i,j])
            i = i - 1
            j = j - 1
         
        elif i > 0 and (current_score == A[j][i - 1] + mismatch or current_score == A[j][i - 1] + gap):
            align_X = [x[i - 1]] + align_X
            align_Y = [0] + align_Y
            i = i - 1
             
        else:
            align_X = [0] + align_X
            align_Y = [y[j - 1]] + align_Y
            j = j - 1
   
    return align_X, align_Y, opt_loc

def local_align(x, y, gap, match, mismatch):
   
    A = []
    for i in range(len(y) + 1):
        A.append([0] * (len(x) +1))
    best = 0
    optloc = [0,0]

    for i in range(1, len(y) + 1):
        for j in range(1, len(x) + 1):
           
            A[i][j] = max(
            A[i][j-1] + gap,
            A[i-1][j] + gap,
            A[i-1][j-1] + (match if y[i-1] == x[j-1] else mismatch),
            0
            )
           
            if A[i][j] >= best:
                best = A[i][j]
                optloc = [i,j]

    align_X = []
    align_Y = []
    j, i = optloc
    opt_loc = []
    while (i > 0 or j > 0) and A[j][i] > 0:
         
        current_score = A[j][i]

        if i > 0 and j > 0 and x[i - 1] == y[j - 1]:
            align_X = [x[i - 1]] + align_X
            align_Y = [y[j - 1]] + align_Y
            opt_loc.append([i,j])
            i = i - 1
            j = j - 1
         
        elif i > 0 and (current_score == A[j][i - 1] + mismatch or current_score == A[j][i - 1] + gap) and A[j][i - 1] > 0:
            align_X = [x[i - 1]] + align_X
            align_Y = [0] + align_Y
            i = i - 1
             
        else:
            align_X = [0] + align_X
            align_Y = [y[j - 1]] + align_Y
            j = j - 1
#    if len(opt_loc)>=5:
#        print(align_X, align_Y, len(opt_loc))
    return align_X, align_Y, opt_loc

#def local_align(x, y, gap, match, mismatch):
#    """Do a local alignment between x and y"""
## create a zero-filled matrix
#    A = np.zeros((len(x) + 1, len(y) + 1)) #make_matrix(len(x) + 1, len(y) + 1)
#    best = 0
#    best_i = 0
#    best_j = 0
##    opt_loc = np.zeros((0,2)) #(0,0)
## fill in A in the right order
#    for i in xrange(1, len(x)+1):
#        for j in xrange(1, len(y)+1):
## the local alignment recurrance rule:
#            A[i][j] = max(A[i][j-1] + gap,\
#                     A[i-1][j] + gap,\
#                     A[i-1][j-1] + (match if x[i-1] == y[j-1] else mismatch))#,\
##                     0)
## track the cell with the largest score
#            if A[i][j] > best:
#                best = A[i][j]
#                best_i = i
#                best_j = j
##                opt_loc = np.vstack((opt_loc,np.array([i,j])))
#
## return the opt score and the best location
#    i = best_i #np.where(A==best)[0][0] #
#    j = best_j #np.where(A==best)[1][0] #
#    opt_loc = np.array([i-1,j-1])
#    score = [A[i][j]]
##    print(i,j)
#    
#    while i > 1 and j > 1:
#        last = np.argmax(np.array([A[i][j-1], A[i-1][j], A[i-1][j-1]+0.01]))
#        if last == 0:
#            opt_loc = np.vstack((opt_loc, np.array([i-1, j-2])))
#            score.append(A[i][j-1])
#            j = j-1
#        elif last == 1:
#            opt_loc = np.vstack((opt_loc, np.array([i-2, j-1])))
#            score.append(A[i-1][j])
#            i = i-1
#        elif last == 2:
#            opt_loc = np.vstack((opt_loc, np.array([i-2, j-2])))
#            score.append(A[i-1][j-1])
#            i = i-1
#            j = j-1
#
#    score = np.array(score)
#    idx = np.where(score>0)[0]
#    score = score[idx]
#    opt_loc = opt_loc[idx]
#    
##    diff = score[:-1]-score[1:]
##    if diff.shape[0]==0: 
##        opt_loc = np.array([])
##    else:
##        idx = np.concatenate(([0], np.where(diff>=10)[0]+1))
##        opt_loc = opt_loc[idx,:]
##        score = score[idx]       
##    
##        diffx = opt_loc[:-1,0]-opt_loc[1:,0]  
##        diffy = opt_loc[:-1,1]-opt_loc[1:,1]        
##        intersec = np.intersect1d(np.where(diffx>0)[0], np.where(diffy>0)[0])
##        idx = np.concatenate(([0], intersec+1))
##        opt_loc = opt_loc[idx,:]
##        score = score[idx]
##    seq_diff = []
##    if opt_loc.shape[0]>=1:
##        for i in range(opt_loc.shape[0]):
##            seq_diff.append(y[opt_loc[i,1]]-x[opt_loc[i,0]]) 
#    return A, score, best, opt_loc