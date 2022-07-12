#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import wfdb
import os

pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings('ignore')

# Phase 2a: Data Collection 

data_files = ["mit-bih-arrhythmia-database-1.0.0/" + file for file in os.listdir("mit-bih-arrhythmia-database-1.0.0") if ".dat" in file]

def read_file(file, participant):
    """Utility function
    """
    # Get signal
    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 0]})
    data["Participant"] = "MIT-Arrhythmia_%.2i" %(participant)
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = 360
    data["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"

    # getting annotations
    anno = wfdb.rdann(file[:-4], 'atr')
    
    #get annotation type for each peak
    check = np.in1d(anno.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])
    anno_type = anno.symbol
    check = pd.DataFrame(check)
    check.rename( columns={0 :'check'}, inplace=True)
    anno_type = pd.DataFrame(anno_type)
    anno_type.rename( columns={0 :'type'}, inplace=True)
    con = pd.concat([check, anno_type], axis=1)
    anntype = con[con.check != False]
    anntype.drop(['check'], inplace=True, axis=1)
    anntype = anntype['type'][:].tolist()
    
    anno = np.unique(anno.sample[np.in1d(anno.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])])
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = "MIT-Arrhythmia_%.2i" %(participant)
    anno["Sampling_Rate"] = 360
    anno["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"
    anno["Type"] = ""
    index=0;
    for x in anntype:
        anno.at[index, 'Type'] = str(x)
        index+=1
    return data, anno


dfs_ecg = []
dfs_rpeaks = []

for participant, file in enumerate(data_files):

    print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))

    data, anno = read_file(file, participant)
    # Store with the rest
    dfs_ecg.append(data)
    dfs_rpeaks.append(anno)

    # Store additional recording if available
    if "x_" + file.replace("mit-bih-arrhythmia-database-1.0.0/", "") in os.listdir("mit-bih-arrhythmia-database-1.0.0/x_mitdb/"):
        print("  - Additional recording detected.")
        data, anno = read_file("mit-bih-arrhythmia-database-1.0.0/x_mitdb/" + "x_" + file.replace("mit-bih-arrhythmia-database-1.0.0/", ""), participant)
        # Store with the rest
        dfs_ecg.append(data)
        dfs_rpeaks.append(anno)

# Save
df_ecg = pd.concat(dfs_ecg).to_csv("ECGs.csv", index=False)
df_rpeaks = pd.concat(dfs_rpeaks).to_csv("Rpeaks.csv", index=False)

###############################################################################
# # Phase 2b: Data Quality Check
# Remove signal noise

import neurokit2 as nk
clean_ecgs = [];
for x in range(len(dfs_ecg)): 
    # Extract clean ECG and R-peaks location
    signals, info = nk.ecg_process((dfs_ecg[x])["ECG"], sampling_rate=360)
    clean_ecgs.append(signals["ECG_Clean"])

# Segment signal into individual heartbeats
heartbeats = [];
for x in range(len(clean_ecgs)): 
    # Segment all the heart beats
    epochs = nk.ecg_segment(clean_ecgs[x], rpeaks=None, sampling_rate=360, show=False)
    temp= pd.concat(epochs)
    temp["Type"] = ""
    temp["Record"] = 0
    temp["Participant"] = ""
    temp = temp.astype({'Label': int,
                        'Record': int,
                        'Index': int
                       })
    temp =  temp.reset_index().drop(['level_0','level_1'], axis=1) # drop multiIndex
    heartbeats.append(temp)

nsymbol = ["N", "L", "R", "B", "e", "j"]
import copy
copy_hb = copy.deepcopy(heartbeats) # use copy

# Assign type of each individual heartbeat, record number, rpeaks
for x in range(len(heartbeats)):
    label_count = copy_hb[x].Label.unique()
    copy_hb[x]['dfs_rpeaks.Label'] = 0
    for y in range (len(label_count)):
        matched = False
        index_num = copy_hb[x].loc[copy_hb[x].Label == (y+1), 'Index'].unique()
        for z in range(len(dfs_rpeaks[x])):
            if((index_num[0] < dfs_rpeaks[x].at[z, 'Rpeaks']) and (dfs_rpeaks[x].at[z, 'Rpeaks'] < index_num[-1])):
                if((dfs_rpeaks[x].at[z,'Type']) not in nsymbol):
                    copy_hb[x].loc[copy_hb[x].Label == (y+1), 'Type'] = "A"
                else:
                    copy_hb[x].loc[copy_hb[x].Label == (y+1), 'Type'] = "N"
                
#                 copy_hb[x].loc[copy_hb[x].Label == (y+1), 'dfs_rpeaks.Label'] = z+1
                copy_hb[x].loc[copy_hb[x].Label == (y+1), 'Participant'] = dfs_rpeaks[x].at[z, 'Participant']
                copy_hb[x].loc[copy_hb[x].Label == (y+1), 'Record'] = x+1
                matched = True
        if(not matched):
            copy_hb[x].drop(copy_hb[x].index[copy_hb[x]['Label'] == (y+1)], inplace=True)
    copy_hb[x] = copy_hb[x].reset_index(drop=True)
    
# Reindex the Index and Label column
for x in range(len(copy_hb)):
    copy_hb[x]['Index'] = pd.Series(range(0, copy_hb[x].shape[0]))
    label_count = copy_hb[x].Label.unique()
    for y in range (len(label_count)):
        copy_hb[x].loc[copy_hb[x].Label == (label_count[y]), 'Label'] = y+1

###############################################################################
# # # Phase 2c: Identify fiducial points
# # Find peaks with neurokit
waves_peak = []
for x in range (len(copy_hb)):
    _, rpeaks = nk.ecg_peaks(copy_hb[x]['Signal'], sampling_rate=360)
    
    # Delineate the ECG signal with DWT
    _, wp = nk.ecg_delineate(copy_hb[x]['Signal'], rpeaks, sampling_rate=360, method="dwt")
    temp = pd.DataFrame(wp)
    temp['Label'] = 0
    temp['Type'] = "N/A"
    temp['Record'] = x+1
    temp['Participant'] = copy_hb[x][copy_hb[x]['Record']==x+1]['Participant']
    temp = temp.astype({'Label': int,
                        'Record': int
                       })
    # Insert Rpeaks
    temp.insert(5, 'ECG_R_Peaks', rpeaks['ECG_R_Peaks'])
    waves_peak.append(temp)

copy_wp = copy.deepcopy(waves_peak)
compare = copy.deepcopy(copy_hb)

# Label each heartbeat id & type in peaks location file
for x in range (len(copy_wp)):
    for y in range (len(copy_wp[x])):
        label_count = compare[x].Label.unique()
        p_onset = copy_wp[x].at[y,'ECG_P_Onsets']
        t_offset = copy_wp[x].at[y,'ECG_T_Offsets']
        if(np.isnan(p_onset) or np.isnan(t_offset)):
            continue
        else:
            for z in label_count:
                index_num = compare[x].loc[compare[x].Label == z, 'Index'].unique()
                if(p_onset >= index_num[0] and t_offset <= index_num[-1]): 
                    copy_wp[x].at[y,'Label'] = z
                    copy_wp[x].at[y,'Type'] = compare[x][compare[x]['Label']==z].Type.values[0]
                    compare[x].drop(compare[x].index[compare[x]['Label'] == z], inplace=True)
                    break

# Filter and remove heartbeats with NaN, or wrong S & T peak detection
for x in range (len(copy_wp)):
    wrong_peak = copy_wp[x][(copy_wp[x]['ECG_S_Peaks'] >= copy_wp[x]['ECG_T_Peaks'])].index
    copy_wp[x].drop(wrong_peak, inplace = True) # drop wrong peak detection
    no_matched = copy_wp[x][(copy_wp[x]['Label'] == 0)].index
    copy_wp[x].drop(no_matched, inplace = True) # drop no matched heartbeats
    copy_wp[x] = copy_wp[x].dropna() # drop NaN
    copy_wp[x] = copy_wp[x].reset_index(drop=True)
    print(x+1, ': -------------------------------------------------')
    print(copy_wp[x])

# Categorize into normal and abnormal heartbeats
dfs_peaks = pd.concat(copy_wp)
dfs_peaks = dfs_peaks.reset_index(drop=True)
grouped = dfs_peaks.groupby("Type")
dfs_abnormal = grouped.get_group("A")
normal = grouped.get_group("N")

# cut down sample size of normal heartbeats
import random
dfs_normal = normal.sample(n=len(dfs_abnormal))

#sort data in asc order
dfs_normal.sort_values(by=['Record', 'Label'], inplace=True)
dfs_normal = dfs_normal.reset_index(drop=True)
dfs_abnormal.sort_values(by=['Record', 'Label'], inplace=True)
dfs_abnormal = dfs_abnormal.reset_index(drop=True)

# Update peaks dataframe
dfs_peaks =  pd.concat([dfs_normal,dfs_abnormal], axis=0)
dfs_peaks.sort_values(by=['Record', 'Label'], inplace=True)
dfs_peaks = dfs_peaks.reset_index(drop=True)

temp_heartbeats = pd.concat(copy_hb)

dfs_heartbeats = pd.DataFrame()
record_count = dfs_peaks.Record.unique()
record_count.sort()
for x in record_count:
    label_count =  dfs_peaks[(dfs_peaks['Record'] == x)].Label.unique()
    label_count.sort()
    temp = temp_heartbeats[(temp_heartbeats['Record'] == x)]
    temp = temp[temp['Label'].isin(label_count)]
    temp.sort_values(by=['Index'], inplace=True)
    dfs_heartbeats = dfs_heartbeats.append(temp, ignore_index = True)


# Save
df_heartbeats = dfs_heartbeats.to_csv("Heartbeats.csv", index=False) # ECG signal of each heartbeats (filtered)
df_peaks = dfs_peaks.to_csv("Peaks.csv", index=False) # all detected peaks location index
df_normal = dfs_normal.to_csv("Normal.csv", index=False) # normal beats
df_abnormal = dfs_abnormal.to_csv("Abnormal.csv", index=False) # abnormal beats
backup_heartbeats = temp_heartbeats.to_csv("backup_Heartbeats.csv", index=False) # ECG signal of each heartbeats (raw)

###############################################################################
# # Phase 2d: Signal trimmering
# Process heartbeats to fixed dimension
record_count = updated_heartbeats.Record.unique()
record_count.sort()
df_cols = ['Signal', 'Label', 'Record' , 'Type', 'Participant']

# Trim signals started from P onset & ended at T offset
trimmed_heartbeats = pd.DataFrame(columns=df_cols);
for x in record_count:
    label_count =  updated_heartbeats[(updated_heartbeats['Record'] == x)].Label.unique() 
    label_count.sort()
    hb = updated_heartbeats[(updated_heartbeats['Record'] == x)]
    hb = hb[hb['Label'].isin(label_count)]
    p = dfs_peaks[(dfs_peaks['Record'] == x)]
    p = p[p['Label'].isin(label_count)]
    for y in label_count:
        hb2 = hb[(hb['Label'] == y)]
        p_on = p[(p['Label'] == y)]['ECG_P_Onsets'].values[0]
        t_off = p[(p['Label'] == y)]['ECG_T_Offsets'].values[0]
        hb2 =  hb2.loc[(hb['Index'] >= p_on) & (hb['Index'] <= t_off)]
        trimmed_heartbeats = trimmed_heartbeats.append(hb2)
trimmed_heartbeats = trimmed_heartbeats.reset_index(drop=True)

# Check length of each heartbeat * picked 250 at the end *
df_cols = ['Length', 'Label', 'Record' , 'Type', 'Participant']
len_heartbeats = pd.DataFrame(columns=df_cols);
for x in record_count:
    label_count =  trimmed_heartbeats[(trimmed_heartbeats['Record'] == x)].Label.unique() 
    label_count.sort()
    temp = trimmed_heartbeats[(trimmed_heartbeats['Record'] == x)]
    temp = temp[temp['Label'].isin(label_count)]
    for y in label_count:
        temp2 = temp[(temp['Label'] == y)]
        len_heartbeats = len_heartbeats.append({
        'Length': len(temp2.Signal),
        'Label': temp2['Label'].values[0],
        'Record': temp2['Record'].values[0],
        'Type': temp2['Type'].values[0],
        'Participant': temp2['Participant'].values[0]
        }, ignore_index = True)
len_heartbeats

# Cut each heartbeat at 250
record_count = trimmed_heartbeats.Record.unique()
record_count.sort()

prep_hb = []

for x in record_count:
    label_count =  trimmed_heartbeats[(trimmed_heartbeats['Record'] == x)].Label.unique() 
    label_count.sort()
    temp = trimmed_heartbeats[(trimmed_heartbeats['Record'] == x)]
    temp = temp[temp['Label'].isin(label_count)]
    for y in label_count:
        temp2 = temp[(temp['Label'] == y)]['Signal']
        temp2 = temp2.values[0:249].tolist()
        prep_hb.append(temp2)

from keras.utils.data_utils import pad_sequences
# pad sequence with zeroes for those length < 250
padded = pad_sequences(prep_hb, padding='post',maxlen=250, dtype='float64')

prep_x = pd.DataFrame(padded)

df_cols = ['Type', 'Label', 'Record', 'Participant']
prep_y = pd.DataFrame(columns=df_cols)

for x in record_count:
    label_count =  trimmed_heartbeats[(trimmed_heartbeats['Record'] == x)].Label.unique() 
    label_count.sort()
    temp = trimmed_heartbeats[(trimmed_heartbeats['Record'] == x)]
    temp = temp[temp['Label'].isin(label_count)]
    for y in label_count:
        temp2 = temp[(temp['Label'] == y)]
        prep_y = prep_y.append({
        'Type': temp2['Type'].values[0],
        'Label': temp2['Label'].values[0],
        'Record': temp2['Record'].values[0],
        'Participant': temp2['Participant'].values[0]
        }, ignore_index = True)

# Join x and y together for final dataset
full_prep_set = prep_x.join(prep_y)

# Save
x = prep_x.to_csv("prep_x.csv", index=False) # ecg signal matrix
y = prep_y.to_csv("prep_y.csv", index=False) # type of each heartbeats ,label, record, participant
full = full_prep_set.to_csv("full_prep_set.csv", index=False) # full data set

