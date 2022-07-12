#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###############################################################################
# Phase 3: Data Classification - Manual Classification

import pandas as pd
dfs_normal = pd.read_csv("Normal.csv")
dfs_abnormal = pd.read_csv("Abnormal.csv")

# Calculation of each ECG parameter - Normal
df_cols = ['PR', 'QRS', 'QT', 'Label', 'Record' , 'Type', 'Participant']
normal_params = pd.DataFrame(columns=df_cols)
frequency = 360
for index, row in dfs_normal.iterrows():
    # Get P onset, R onset, R offset, T onset
    p_onset = dfs_normal.at[index,'ECG_P_Onsets']
    r_onset = dfs_normal.at[index,'ECG_R_Onsets']
    r_offset = dfs_normal.at[index,'ECG_R_Offsets']
    t_offset = dfs_normal.at[index,'ECG_T_Offsets']

    # PR interval: P onsets -> R onsets
    pr = round(((r_onset - p_onset) / frequency), 2)
    # QRS complex: R onsets -> R offsets
    qrs = round(((r_offset - r_onset) / frequency),2)
    # QT segment: R onsets -> T offsets
    qt = round(((t_offset - r_onset) / frequency),2)
    
    temp_list = [pr, qrs, qt]
    if(all(x > 0 for x in temp_list)):
        normal_params = normal_params.append({
        'PR': pr,
        'QRS': qrs,
        'QT': qt,
        'Label': dfs_normal.at[index,'Label'],
        'Record': dfs_normal.at[index,'Record'],
        'Type': 'N',
        'Participant': dfs_normal.at[index,'Participant']
        }, ignore_index = True)
        
# Calculation of each ECG parameter - Abnormal
abnormal_params = pd.DataFrame(columns=df_cols)
frequency = 360
for index, row in dfs_abnormal.iterrows():
    # Get P onset, R onset, R offset, T onset
    p_onset = dfs_abnormal.at[index,'ECG_P_Onsets']
    r_onset = dfs_abnormal.at[index,'ECG_R_Onsets']
    r_offset = dfs_abnormal.at[index,'ECG_R_Offsets']
    t_offset = dfs_abnormal.at[index,'ECG_T_Offsets']

    # PR interval: P onsets -> R onsets
    pr = round(((r_onset - p_onset) / frequency), 2)
    # QRS complex: R onsets -> R offsets
    qrs = round(((r_offset - r_onset) / frequency),2)
    # QT segment: R onsets -> T offsets
    qt = round(((t_offset - r_onset) / frequency),2)
    
    temp_list = [pr, qrs, qt]
    if(all(x > 0 for x in temp_list)):
        abnormal_params = abnormal_params.append({
        'PR': pr,
        'QRS': qrs,
        'QT': qt,
        'Label': dfs_abnormal.at[index,'Label'],
        'Record': dfs_abnormal.at[index,'Record'],
        'Type': 'A',
        'Participant': dfs_abnormal.at[index,'Participant']
        }, ignore_index = True)

all_params = pd.concat([normal_params,abnormal_params], ignore_index=True)

#  # Classify heartbeats based on reference 
# PR: >= 0.12 AND <= 0.22
# QRS <= 0.12
# QT <= 0.47

result = []
for index, row in all_params.iterrows():
    pr = all_params.at[index,'PR']
    qrs = all_params.at[index,'QRS']
    qt = all_params.at[index,'QT']
    if((0.12 <= pr <= 0.22) and (qrs <= 0.12) and (qt <= 0.47)):
        result.append('N')
    else:
        result.append('A')
all_params = all_params.assign(Result=result)

# Save
ab_params = abnormal_params.to_csv("abnormal_params.csv", index=False) # ECG parameters of abnormal beats
n_params = normal_params.to_csv("normal_params.csv", index=False) # ECG parameters of normal beats
params = all_params.to_csv("all_params.csv", index=False) # ECG parameters of all

