# In[1]:

import os
import subprocess
import numpy as np
np.get_include()

# In[2]:

## Scripts containing inference algorithms
from BaumWelch import *
from BaumWelchLR import *
from TVAR import *

# In[3]:

import numpy as np
import pandas as pd
import csv
#from numpy import linspace,exp
#from numpy.random import randn
#import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
#import seaborn as sns

import scipy 
import editdistance
import sklearn.metrics
import statsmodels.api as sm

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


## Helper functions for working with note pitch representation

## Convert from pitch representation (integers 0-127) to integers (0-max)
## x is the input vector of notes and code is a vector of the unique pitches in x
def encode(x, code):
    output = np.array([int(np.where(code == x[i])[0]) for i in range(0, len(x))])
    return output


## Reverses the function encode
## x is the vector of pitches to decode and code is a vector of the unique pitches in x before it was encoded
def decode(x, code):
    output = np.zeros(len(x))
    for i in range(0, len(x)):
        output[i] = code[x[i]]
    return output

## Function to convert the values in array to the nearest values in the array value
## Used to convert continues TVAR generated pitches to closest integer values for MIDI representation
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


# # Metrics Functions

# In[5]:


## Function to create a matrix of notes and times representing the input piece as a matrix
## time is the time steps (integers) at which a note occurs
## notes are the note pitches (integers 0-127) where each note is "turned on" and "turned off"
## velocity is the note velocity (i.e. volume) at each time step
## measures is the number measures in the original piece
## min_note is the length of the shortest note in the original piece, using same scale as time array above
## num is the number of quarter notes in a measure (i.e. represents the numerator of time signature of the input piece)
## met_mat output is a matrix where columns correspond to the time stamp of notes, one column for each min_note
##    duration for the entire piece and the rows are the note pitches, values in the matrix are 1 for the duration
##    of a note when it is played and 0 when it is not played
def create_matrix(time, notes, velocity, measures, min_note, num):
    met_mat = pd.DataFrame(np.zeros(shape = (len(np.unique(notes)), int(measures)*num), dtype = int))
    met_mat.index = np.unique(notes)[::-1]
    met_mat.columns = np.arange(0, min_note*num*measures, min_note)[:int(measures)*num]
    max_time = met_mat.columns[-1]
    for i in np.unique(notes):
        on = time[np.intersect1d(np.where(notes == i), np.where(velocity > 0) )]
        off = time[np.intersect1d(np.where(notes == i), np.where(velocity == 0) )]
        if len(off) % 2 !=0 or len(on) %2 !=0:
            off = np.append(off, max_time)
        for j in range(len(on)):
            met_mat.loc[i, on[j]:off[j]] = 1
    return(met_mat)

## Function to calculate the musical metrics of generated pieces
## met_mat is the output from create_matrix()
## harmonic ints is a vector of length 12 corresponding to the counts of each type of harmonic interval in the piece considered
## melodic ints is a vector of length 12 corresponding to the counts of each type of melodic interval in the piece considered
## percentage is a vector of length 6 containing the percentage of perfect harmonic intervals, the percentage of imperfect 
##           consonant harmonic intervals, the percentage of dissonant harmonic intervals, the percentage of perfect melodic
##           intervals, the percentage of imperfect consonant melodic intervals and the percentage of dissonant melodic intervals
def musical_metrics(met_mat):
    perfect = np.array([0,5,7])
    imperfect = np.array([3,4,8,9])
    dissonant = np.array([1,2,6,10,11])
    major_scale = np.array([2,2,1,2,2,2,1])
    harmonic_ints = np.zeros(12)
    c = 0
    max_notes = np.max(np.sum(met_mat, axis = 0))
    melodic = np.zeros(shape = (max_notes, len(met_mat.columns)))
    for col in met_mat.columns:
        chord = np.array(met_mat.index[np.where(met_mat[col] == 1)[0]])[::-1]
        if len(chord) > 0:
            intervals = np.diff(chord)
            intervals[intervals >= 12] = intervals[intervals >= 12] % 12
            harmonic_ints[intervals.astype(int)] +=1
            melodic[:len(chord), c] = chord
            c+= 1
    melodic_ints = list()
    for t in range(melodic.shape[1] - 1):
        m1 = melodic[melodic[:,t] > 0, t]
        m2 = melodic[melodic[:,t+1] > 0, t+1]
        melodic_ints.append(np.unique([abs(i-j) %12 for i in m1 for j in m2]))

    u = np.unique(np.hstack(melodic_ints), return_counts=True)
    m_ints = np.zeros(12)
    m_ints[u[0].astype(int)] = u[1].astype(int)

    h_total = np.sum(harmonic_ints)
    m_total = np.sum(m_ints)

    h_per = np.sum(harmonic_ints[perfect])/h_total
    h_imp = np.sum(harmonic_ints[imperfect])/h_total
    h_dis = np.sum(harmonic_ints[dissonant])/h_total

    m_per = np.sum(m_ints[perfect])/m_total
    m_imp = np.sum(m_ints[imperfect])/m_total
    m_dis = np.sum(m_ints[dissonant])/m_total

    percentage = np.array([h_per, h_imp, h_dis, m_per, m_imp, m_dis])

    return(harmonic_ints, m_ints, percentage)

## Calculate the empirical entropy of the input data and output as a vector in entropy
def ent(data):
    p_data= np.unique(data, return_counts = True)[1]/len(data) # calculates the probabilities
    entropy=scipy.stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy


## Function to compare an original piece to a generated piece and calculate metrics
## old_notes is a vector of the original piece's note pitches
## new_notes is a vector of the generated piece's note pitches
## Returns the empirical entropy, mutual information and edit distance between the original piece and the new, generated piece
##        also returns the count of unique notes in the generated piece, normalized by the total number of notes
def originality_metrics_comparison(old_notes, new_notes):
    # Calculate entropy
    entropy = ent(new_notes)

    # Calculate edit distance
    edit_dist = editdistance.eval(old_notes, new_notes)/len(old_notes)

    # Calculate mutual info
    mutual_info = sklearn.metrics.mutual_info_score(old_notes, new_notes)

    k = len(np.unique(old_notes))
    possibleNotes = np.unique(old_notes)    
    # Calculate note counts
    unique_new_notes, note_counts = np.unique(new_notes, return_counts = True)

    if len(unique_new_notes) != k:
        add_notes = list(set(possibleNotes) - set(unique_new_notes))
        for i in add_notes:
            if np.where(possibleNotes == i)[0] > len(note_counts):
                note_counts = np.append(note_counts, np.where(possibleNotes == i)[0], 0)
            else:
                note_counts = np.insert(note_counts, np.where(possibleNotes == i)[0], 0)
    note_counts = note_counts/len(old_notes)
    return(entropy, mutual_info, edit_dist, note_counts)


## Function to calculate the ACF and PACF out to lag 40
## new_ntoes is the input vector of note pitches
## note_acf is a vector of length 41 of the ACF values and note_pacf is a vector of length 41 of the PACF values
def time_metrics(new_notes):   
    #Calculate ACF/PACF out to lag 40
    note_acf = sm.tsa.stattools.acf(new_notes)
    try:
        note_pacf = sm.tsa.stattools.pacf(new_notes)
    except np.linalg.linalg.LinAlgError as err:
        note_pacf = sm.tsa.stattools.pacf(new_notes)

    return(note_acf, note_pacf)


## Function to calculate all metrics
## time is the time steps (integers) at which a note occurs
## notes are the note pitches (integers 0-127) where each note is "turned on" and "turned off"
## velocity is the note velocity (i.e. volume) at each time step
## measures is the number measures in the original piece
## min_note is the length of the shortest note in the original piece, using same scale as time array above
## num is the number of quarter notes in a measure (i.e. represents the numerator of time signature of the input piece)
## output is a vector of the calculated metrics:
##        entropy is the empirical entropy of new_notes
##        mutual_info is the mutual information between old_notes and new_notes
##        edit_dist is the edit distance between old_notes and new_notes
##        harmonic_ints is a vector of length 12 of the count of harmonic intervals of each type in new_notes
##        melodic_ints is a vector of length 12 of the count of melodic intervals of each type in new_notes
##        percentage is a vector of length 6 containing the percentage of perfect harmonic intervals, the percentage of imperfect 
##           consonant harmonic intervals, the percentage of dissonant harmonic intervals, the percentage of perfect melodic
##           intervals, the percentage of imperfect consonant melodic intervals and the percentage of dissonant melodic intervals      
##       note_counts is a vector of length equal to the number of unique pitches in old_notes with a normalized count of pitches
##           in new_notes
##       note_acf is a vector of length 41 with the acf of new_notes
##       note_pacf is a vector of length 41 with the pacf of new_notes
## Note: time is the same for old_notes and new_notes, as this is not changed between the original and generated pieces
##  (likewise, measures, min_note and num are the same for old_notes and new_notes)
def calc_metrics(time, old_notes, new_notes, velocity, measures, min_note, num):
    met_mat = create_matrix(time, new_notes, velocity, measures, min_note, num)
    harmonic_ints, m_ints, percentage = musical_metrics(met_mat)
    entropy, mutual_info, edit_dist, note_counts = originality_metrics_comparison(old_notes, new_notes)
    note_acf, note_pacf = time_metrics(new_notes)
    return(np.hstack((np.array([entropy, mutual_info, edit_dist]), 
           harmonic_ints, m_ints, percentage, note_counts, note_acf, note_pacf)))


# ## Generate New Pieces
# 
# Functions to take learned parameters and generate new pieces

# In[6]:


## Function to generate new pieces from the HMM, 2-HMM, 3-HMM, LR-HMM, 2LR-HMM, and 3-LR HMM
## n is the length of the original and generated piece
## pi is the learned initial distribution
## phi is the learned emission distribution
## Tmat is the learned transition matrix
## T2mat is the learned second order transition matrix (if applicable)
## T3mat is the learned third order transition matrix (if applicable)
## code is the unique note pitches occurring in the original piece
## model is the model order which the input parameters correspond to, either "first_order", "second_order" or "third_order"
## Outputs: output is the note pitches of the generated pieces (array of length n), z are the generated hidden states 
##          (vector of length n)
def hmm(n, pi, phi, Tmat, T2mat, T3mat, code, model):
    m = Tmat.shape[0]
    k = phi.shape[1]
    zstates = np.arange(0, m, dtype = int)
    xstates = np.arange(0, k, dtype = int)
    z = np.zeros(n, dtype = int)
    x = np.zeros(n, dtype = int)
    z[0] = np.random.choice(zstates, size = 1, p = pi)
    if model == 'first_order':
        for j in range(1, n):
            z[j] = np.random.choice(zstates, size = 1, p = Tmat[z[j-1], :])
        for i in range(0, n):
            x[i] = np.random.choice(xstates, size = 1, p = phi[z[i], :])
     
    if model == 'second_order':
        z[1] = np.random.choice(zstates, size = 1,  p = Tmat[z[0], :])
        for j in range(2, n):
            z[j] = np.random.choice(zstates, size = 1,  p = T2mat[z[j-2],z[j-1], :])
        for i in range(0, n):
            x[i] = np.random.choice(xstates, size =1, p = phi[z[i], :])
    if model == 'third_order':
        z[1] = np.random.choice(zstates, size = 1,  p = Tmat[z[0], :])
        z[2] = np.random.choice(zstates, size = 1,  p = T2mat[z[0],z[1], :])
        for j in range(3, n):
            z[j] = np.random.choice(zstates, size = 1,  p = T3mat[z[j-3],z[j-2],z[j-1], :])
        for i in range(0, n):
            x[i] = np.random.choice(xstates, size =1, p = phi[z[i], :])
    output = decode(x, code)
    return (output, z)

## Function to generate new pieces for TSHMM
## n is the length of the original and generated piece
## pi is the learned initial distribution
## phi is the learned emission distribution
## Tmat is the learned transition matrix
## A is the transition matrix for the R hidden states
## B is the matrix to generate the S hidden states
## code is the unique note pitches occurring in the original piece
## Outputs: output is the note pitches of the generated pieces (array of length n), S and R are the generated hidden states 
##          (vector of length n)
def hmm_2hidden(n, pi, phi, Tmat, A, B, code):
    N = A.shape[0]
    M = B.shape[1]
    k = phi.shape[1]
    zstates = np.arange(0, N*M)
    rstates = np.arange(0,N)
    sstates = np.arange(0,M)
    xstates = np.arange(0, k)
    R = np.zeros(n, dtype = int)
    S = np.zeros(n, dtype = int)
    x = np.zeros(n, dtype = int)
    z = np.random.choice(zstates, size = 1, p = pi)
    S[0] = z % M
    R[0] = int((z - z%M)/M)
    for j in range(1, n):
        R[j] = np.random.choice(rstates, size = 1, p = A[R[j-1], :])
        S[j] = np.random.choice(sstates, size = 1, p = B[R[j], S[j-1],:])
    for i in range(0, n):
        x[i] = np.random.choice(xstates, size = 1, p = phi[S[i], :])
    output = decode(x, code)
    return(output, [S, R])

## n is the length of the original and generated piece
## pi is the learned initial distribution
## phi is the learned emission distribution
## Tmat is the learned transition matrix
## psi is the transition matrix between observed states
## code is the unique note pitches occurring in the original piece
## model is the model order which the input parameters correspond to, either "first_order", "second_order" or "third_order"
## Outputs: output is the note pitches of the generated pieces (array of length n), z are the generated hidden states 
##          (vector of length n)
def hmmARHMM(n, pi, phi, Tmat, psi, code):
    m = Tmat.shape[0]
    k = phi.shape[1]
    zstates = np.arange(0, m)
    xstates = np.arange(0, k)
    z = np.zeros(n, dtype = int)
    x = np.zeros(n, dtype = int)
    z[0] = np.random.choice(zstates, size = 1, p = pi)
    x[0] = np.random.choice(xstates, size = 1, p = phi[z[0], :])
    
    for j in range(1, n):
        z[j] = np.random.choice(zstates, size = 1, p = Tmat[z[j-1], :])
    for i in range(0, n):
        x[i] = np.random.choice(xstates, size = 1, p = psi[z[i], :, x[i-1]])
    output = decode(x, code)
    return(output, z)


# # Pre-Processing

# In[7]:


## Function to pre-process input CSV of original song into form that can be used for modeling and metrics
## Assumes original piece's MIDI file has been converted to a CSV using http://www.fourmilab.ch/webtools/midicsv/#midicsv.5
## input_filename = name of original csv
## output_filename = name of new csv to save generated piece to
## Outputs:
##         quarter_note = number of time steps corresponding to 1 quarter note
##         num = numerator in key signature
##         denom = denominator in key signature
##         key = key signature of piece, integer between -7 and 7 where 0 is C Major
##         measures = number of measures in input piece
##         time = vector of time stamps at which notes occurr
##         notes = vector of note pitches (integers 0-127)
##         velocity = "volume" of each note pitch, 0 = note off, length of time is the same as length of notes and velocity
##         song = pandas dataframe to use for output generated pieces, retains formatting expected by MIDI-CSV
##         song.index = index of original song dataframe

## See http://www.fourmilab.ch/webtools/midicsv/#midicsv.5 for a discussion of MIDI and CSV formats

class pre_process(object):
    def __init__(self, input_filename, min_note):
        self.input_filename = input_filename
        self.min_note = min_note
      
    
    def read_process(self):
        with open(self.input_filename,encoding = "ISO-8859-1") as fd:
            reader=csv.reader(fd)
            rows= [row for idx, row in enumerate(reader)]
        song = pd.DataFrame(rows)
        r,c = np.where(song == ' Header')
        quarter_note = song.iloc[r,5].values.astype(int)[0]
        r, c = np.where(song == ' Time_signature')
        num = song.iloc[r, 3].values.astype(int)[0]
        denom = song.iloc[r, 4].values.astype(int)[0]**2
        try:
            r, c = np.where(song == ' Key_signature')
            key = song.iloc[r,3].values.astype(int)[0]
        except:
            key = None
        #assume the max track possible is 3
        song = song[song[0].astype(int) < 4]
            
        #seems to find which rows have the first column equal to the max in that column
        #song_model = song_model.loc[song_model.iloc[:,0] == np.max(song.iloc[:,0])]
        
        song_model = song[song.iloc[:, 2].isin([' Note_on_c', ' Note_off_c'])]
        time = np.array(song_model.iloc[:,1]).astype(int)
        notes = np.array(song_model.iloc[:,4]).astype(int)
        velocity = np.array(song_model.iloc[:,5]).astype(int)
        measures = np.round(np.max(time)/quarter_note)/num
        min_note = quarter_note
        actual = np.arange(0, min_note*measures*num, min_note).astype(int) 
        time = np.array([find_nearest(actual, time[i]) for i in range(len(time))]).astype(int)
        return(quarter_note, num, denom, key, measures, time, notes, velocity, song, song_model.index)


# ## Velocity
# 
# Velocity is not explicitly modeled in this work.  Velocity is the "loudness" of each note pitch, where 0 is the note is off.  The first occurrence of a note pitch turns a note on and the next occurrence turns the same pitch off.  After a new sequence of note pitches was generated, the velocity for turning the pitch on and off was imputed accordingly.  The remaining non-0 velocities were produced using splines of the velocities of the original piece.

# In[8]:


## newNotes = vector of note pitches of new, generated piece
## velocity = velocity of original piece
## newVelocities = velocities for newNotes, with 0s appropriately filled in and spline values for other non-0 values

def find_vel(newNotes, velocity):
    # Use splines to interpolate the velocities
    newVelocities = np.zeros(len(newNotes))
    y = velocity[np.nonzero(velocity)]
    indicies = []
    for i in np.unique(newNotes):
        indicies.append(np.where(newNotes == i)[0][::2])  ## set every other pitch occurrence to 0 (turn off)

    unlist = [item for sublist in indicies for item in sublist]
    unlist.sort()
    X = np.array(range(0,len(y)))
    s = UnivariateSpline(X, y, s=300) #750
    xs = np.linspace(0, len(y), len(unlist), endpoint = True)
    ys = s(xs)    
    newVelocities[np.array(unlist)] = np.round(ys).astype(int)
    #Fix entries that are too small or too large due to spline overfitting
    newVelocities[np.where(newVelocities < 0)[0]] = y[-1]
    newVelocities = newVelocities.astype(int)    
    newVelocities[np.where(newVelocities > 127)[0]] = 127
    return(newVelocities)


# # HMM_compose
# 
# This is the main function to take in an original piece, learn the appropriate model parameters, generate a new piece, calculate metrics and output the results.
# **Inputs:**
# - input_filename = csv file of original piece (converted from MIDI using http://www.fourmilab.ch/webtools/midicsv/#midicsv.5)
# - output_filename = filename for csvs of generated pieces and metrics
# - min_note = length of shortest note occurring in original piece
# - model = appropriate HMM model to fit, options include 'first_order', 'random', 'first_order-LR', 'second_order', 'second_order-LR', 'third_order', 'third_order-LR', 'TSHMM', 'ARHMM', 'HSMM', 'TVAR', 'factorial' and 'layered'
# - m = number of hidden states for model 
# - tol = tolerance for convergence of inference algorithms
# - it = number of generated pieces to produce to calculate metrics
# - m2 = number of hidden states for the top level of the TSHMM
# - metrics_calc = True (calculate metrics) or False (generate piece only)
# - case_study = True (return parameters to explore), False (only save generated piece and metrics to CSV, no other outputs)
# 
# **Outputs:**
# - generated piece to 'output_filename', if multiple pieces are generated for metrics, last generated piece is saved by default
# - metrics for number of generated pieces specified by it are saved to metrics folder
# - If case_study = True:
#     - time = time stamp for each note in the original and generated pieces
#     - notes = original notes
#     - newNotes = generated note pitches
#     - z = sequence of generated hidden states
#     - pi1 = learned initial distribution
#     - phi1 = learned emission distribution
#     - Tmat1 = learned transition distribution
# 
# Note: printed intergers correspond to iteration of inference algorithm

# In[ ]:


def hmm_compose(input_filename, output_filename, min_note, model, m,  tol, it, m2 = None, metrics_calc = False,
               case_study = False):
    quarter_note, num, denom, key, measures, time,             notes, velocity, song, ind = pre_process(input_filename, min_note).read_process()

    #Find possible unique notes and velocities
    possibleNotes = np.unique(notes)
    possibleVelocities =  np.unique(velocity)

    k = len(possibleNotes)
    xNotes = encode(notes, possibleNotes)
    n = len(xNotes)
    

    if metrics_calc:
        orig_metrics = calc_metrics(time, notes, notes, velocity, measures, min_note, num)
        metrics = np.zeros(shape = (it+1, len(orig_metrics)))
        metrics[0,:] = orig_metrics
    

    #Run BaumWelch for specified model
    if model == 'first_order':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        newNotes, z  = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
        newVelocities = find_vel(newNotes, velocity)
        
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)
            
                
                
    if model == 'random':
        vals = np.random.rand(m)
        pi1 = vals/np.sum(vals)
        Tmat1 = np.zeros(shape = (m, m))
        phi1 = np.zeros(shape = (m, k))
        vals1 = np.random.rand(m,m)
        vals2 = np.random.rand(m,k)
        Tmat1 = vals1/np.sum(vals1, axis=1)[:,None]
        phi1 = vals2/np.sum(vals2, axis = 1)[:,None]
        newNotes,z  = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
        newVelocities = find_vel(newNotes, velocity)
        
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)





    if model == 'first_order-LR':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        newNotes,z  = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
        newVelocities = find_vel(newNotes, velocity)
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)



    if model == 'second_order-LR':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        it2, T2mat = second_orderLR(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        newNotes,z  = hmm(n, pi1, phi1, Tmat1, T2mat, None, possibleNotes,'second_order')
        newVelocities = find_vel(newNotes, velocity)
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmm(n, pi1, phi1, Tmat1, T2mat, None, possibleNotes,'second_order')
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)



    if model == 'second_order':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        it2, T2mat = second_order(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        newNotes,z  = hmm(n, pi1, phi1, Tmat1, T2mat, None, possibleNotes,'second_order')
        newVelocities = find_vel(newNotes, velocity)
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmm(n, pi1, phi1, Tmat1, T2mat, None, possibleNotes,'second_order')
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)
        

    if model == 'third_order-LR':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        it2, T2mat = second_orderLR(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        it3, T3mat = third_orderLR(n, m, k, xNotes, pi1, Tmat1, T2mat, phi1, tol)
        newNotes,z  = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat, possibleNotes,'third_order')
        newVelocities = find_vel(newNotes, velocity)
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat, possibleNotes,'third_order')
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)



    if model == 'third_order':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        it2, T2mat = second_orderLR(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        it3, T3mat = third_orderLR(n, m, k, xNotes, pi1, Tmat1, T2mat, phi1, tol)
        newNotes,z  = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat, possibleNotes,'third_order')
        newVelocities = find_vel(newNotes, velocity)
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat, possibleNotes,'third_order')
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)

    if model == 'TSHMM':
        it1, p1, pi1, phi1, Tmat1, A1, B1 = two_hidden_states(n, m, m2, k, xNotes, tol)
        newNotes,z  = hmm_2hidden(n, pi1, phi1, Tmat1, A1, B1, possibleNotes)
        newVelocities = find_vel(newNotes, velocity)
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmm_2hidden(n, pi1, phi1, Tmat1, A1, B1, possibleNotes)
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)


    if model == 'HSMM':
        it1, p1, pi1, phi1, Tmat1 =  HSMM(n, m, k, xNotes, tol)
        newNotes,z  = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
        newVelocities = find_vel(newNotes, velocity)
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)


    if model == 'ARHMM':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        it2, p2, psi = first_orderARHMM(n, m, k, np.log(pi1), np.log(Tmat1), np.log(phi1), xNotes,  tol)
        newNotes,z  = hmmARHMM(n, pi1, phi1, Tmat1, psi, possibleNotes)
        newVelocities = find_vel(newNotes, velocity)
        if metrics_calc:
            for i in range(it):
                newNotes,z  = hmmARHMM(n, pi1, phi1, Tmat1, psi, possibleNotes)
                newVelocities = find_vel(newNotes, velocity)
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)


    if model == 'TVAR':
        # Find parameters that maximize likelihood
        x = notes - np.mean(notes)
        T = n
        pvals=np.array([7, 15]) 
        p=pvals[1]  
        dn=np.arange(0.94, 0.975,.005) 
        bn=np.arange(0.85, 0.915, 0.005) 
        m0=np.zeros(shape = (p,1)); n0=1; s0=0.01; C0=np.identity(p); 
        [popt,delopt,likp] = tvar_lik(x,pvals,dn,bn,m0,C0,s0,n0);
        print(popt)

        # Fit TVAR
        p=popt; m0=np.zeros(shape = (p,1)); n0=1; s0=0.01; C0=np.identity(p);  # initial priors 
        delta=delopt
        [m,C,n,s,e,mf,Cf,sf,nf,ef,qf] = tvar(x,p,delta,m0,C0,s0,n0);

        # Simulate from TVAR
        N=it; # MC sample size
        times=range(T);
        phis = tvar_sim(m,C,n,s,times,N)
        print(phis.shape)

        # Generate new notes

        err_term = np.random.normal(0, np.sqrt(s))
        z = np.zeros(len(notes))
        newNotes = x
        for i in range(it):
            for t in range(p, T):
                if t == p:
                    newNotes[t] = np.dot(x[t-1::-1], phis[:,t,0]) + err_term[t]
                    z[t] = np.dot(x[t-1::-1], phis[:,t,0])
    
                else:
                    newNotes[t] = np.dot(x[t-1:t-p-1:-1], phis[:,t,0]) + err_term[t]
                    z[t] = np.dot(x[t-1:t-p-1:-1], phis[:,t,0])
                    
            newNotes = np.round(newNotes + np.mean(notes))    
            
                
            for j in range(len(notes)):
                if newNotes[j] not in possibleNotes:
                    newNotes[j] = find_nearest(possibleNotes, newNotes[j])
            newVelocities = find_vel(newNotes, velocity)
            if metrics_calc:
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)
    
        phi1 = None
        Tmat1 = None
        pi1 = None

        m = popt

    if model == 'factorial':  #originally 15, 10, 5, but 5,5,5 for case_study
        xstates = range(0, k)
        noteArray = np.zeros(shape = (3, n))
        if case_study:
            it1, p1, pi1, phi15, Tmat1 = first_order(n, 5, k, xNotes, tol)
            zStar15 = Viterbi(n, 5, k, np.log(pi1), np.log(Tmat1), np.log(phi15), xNotes)
            zStar15 = np.array(zStar15).astype(int)
            it1, p1, pi1, phi10, Tmat1 = first_order(n, 5, k, xNotes, tol)
            zStar10 = Viterbi(n, 5, k, np.log(pi1), np.log(Tmat1), np.log(phi10), xNotes)
            zStar10 = np.array(zStar10).astype(int)
            it1, p1, pi1, phi5, Tmat1 = first_order(n, 5, k, xNotes, tol)
            zStar5 = Viterbi(n, 5, k, np.log(pi1), np.log(Tmat1), np.log(phi5), xNotes)
            zStar5 = np.array(zStar5).astype(int)
            z = [zStar15, zStar10, zStar5]
            phi1 = [phi15, phi10, phi5]
        
        else:
            it1, p1, pi1, phi15, Tmat1 = first_order(n, 15, k, xNotes, tol)
            zStar15 = Viterbi(n, 15, k, np.log(pi1), np.log(Tmat1), np.log(phi15), xNotes)
            zStar15 = np.array(zStar15).astype(int)
            it1, p1, pi1, phi10, Tmat1 = first_order(n, 10, k, xNotes, tol)
            zStar10 = Viterbi(n, 10, k, np.log(pi1), np.log(Tmat1), np.log(phi10), xNotes)
            zStar10 = np.array(zStar10).astype(int)
            it1, p1, pi1, phi5, Tmat1 = first_order(n, 5, k, xNotes, tol)
            zStar5 = Viterbi(n, 5, k, np.log(pi1), np.log(Tmat1), np.log(phi5), xNotes)
            zStar5 = np.array(zStar5).astype(int)
            z = [zStar15, zStar10, zStar5]
            phi1 = [phi15, phi10, phi5]

        for i in range(it):
            for j in range(0, n):
                noteArray[0,j] = np.random.choice(xstates, size = 1, p = phi15[zStar15[j], :])
                noteArray[1,j] = np.random.choice(xstates, size = 1, p = phi10[zStar10[j], :])
                noteArray[2,j] = np.random.choice(xstates, size = 1, p = phi5[zStar5[j], :])
            temp_notes = np.rint(np.mean(noteArray, axis=0)).astype(int)
            temp_notes = decode(temp_notes, possibleNotes)
            newNotes = temp_notes
            newVelocities = find_vel(newNotes, velocity)
            if metrics_calc:
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)


    if model == 'layered':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        zStar1 = Viterbi(n, m, k, np.log(pi1), np.log(Tmat1), np.log(phi1), xNotes)
        zStar1 = np.array(zStar1).astype(int)
        it2, p2, pi2, phi2, Tmat2 = first_order(n, m, m, zStar1, tol)
        zStar2 = Viterbi(n, m, m, np.log(pi2), np.log(Tmat2), np.log(phi2), zStar1)
        zStar2 = np.array(zStar2).astype(int)
        it3, p3, pi3, phi3, Tmat3 = first_order(n, m, m, zStar2, tol)
        zStar3 = Viterbi(n, m, m, np.log(pi3), np.log(Tmat3), np.log(phi3), zStar2)
        zStar3 = np.array(zStar3).astype(int)
        output = np.zeros(shape = (3,n), dtype = int)
        z = [zStar1, zStar2, zStar3]
        
        xstates = range(0, k)
        zstates = range(0, m)
        for i in range(it):
            for j in range(0,n):
                output[2, j] = np.random.choice(zstates, size = 1, p = phi3[zStar3[j], :])
                output[1, j] = np.random.choice(zstates, size = 1, p = phi2[output[2, j], :])
                output[0, j] = np.random.choice(xstates, size = 1, p = phi1[output[1, j], :])
            temp_notes = decode(output[0,:], possibleNotes).astype(int)
            newNotes = temp_notes
            newVelocities = find_vel(newNotes, velocity)
            if metrics_calc:
                metrics[i+1, :] = calc_metrics(time, notes, newNotes, newVelocities, measures, min_note, num)
        phi1 = [phi1, phi2, phi3]
        

    song.iloc[ind, 1] = time
    song.iloc[ind, 4] = newNotes
    song.iloc[ind, 5] = newVelocities
    song.iloc[ind[np.where(newVelocities !=0)], 2] = ' Note_on_c'
    song.iloc[ind[np.where(newVelocities ==0)], 2] = ' Note_off_c'
    
    #remove lines with non-note events
    non_note_events = [' Control_c', ' Program_c', ' Pitch_bend_c']
    song = song[~(song[2].isin(non_note_events)) | (song[1] == 0)]
    
    split = output_filename.split('.')
    output_filename = split[0] + '__'+ model + '_' + str(m)+  '-tol' +str(tol)+'.' + split[1]
    if m2 != None:
        output_filename = split[0] + '__'+ model + '_' + str(m)+'-'+str(m2)+ '-tol' +str(tol)+ '.' + split[1]

    if metrics_calc:
        song_name = split[0].split('/')
        metrics_filename = 'metrics/'+song_name[1]+ '__'+ model + '_' + str(m)+  '-tol' +str(tol)+ '.' + split[1] 
        pd.DataFrame(metrics).to_csv(metrics_filename, header = None, index = False)
        print(metrics_filename)
    song.to_csv(output_filename, header = None, index = False, quoting=csv.QUOTE_NONE)
    
    if case_study:
        return(time, notes, newNotes, z, pi1, phi1, Tmat1) #quarter_note, num, denom, key, measures




