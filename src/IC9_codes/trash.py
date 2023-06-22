import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re                                   

#creating DataFrame with all clinical notes
import glob
import os

all_files = glob.glob(os.path.join("/home/ana.lopes/mscmulmiesclin_v0.1", "*.csv"))

df_notes = pd.concat((pd.read_csv(f, index_col = [0]) for f in all_files), ignore_index=True )

#Convert str to list
import ast

df_notes['tokens'] = df_notes['tokens'].apply(lambda x: ast.literal_eval(x))
df_notes['upos'] = df_notes['upos'].apply(lambda x: ast.literal_eval(x))

print(type(df_notes['tokens'][0]))
print(type(df_notes['upos'][0]))



note1 = df_notes['Clinical Text'][3]



def Remove_Discharge_Diagnosis(clinical_text):
    pos1, pos2 = 0,0
    clinical_text = clinical_text.lower()
    if "discharge diagnoses" in clinical_text:
        pos1 = clinical_text.index('discharge diagnoses') 
        for i in range(pos1, len(clinical_text)-1,1):
            if clinical_text[i] == "\n" and clinical_text[i+1]=="\n":
                pos2 = i+1
                break
    discharge_diagnoses = clinical_text[pos1:pos2]
    if len(discharge_diagnoses)==0:
        return None,None
    else:
        indexes = set([j for j in range(pos1, pos2+1, 1)])    
        clinical_text  =  "".join([char for idx, char in enumerate(clinical_text) if idx not in indexes])
        return clinical_text, discharge_diagnoses

df_notes['Clinical Note no diagnoses'], df_notes['Discharge Diagnoses'] =zip(*df_notes['Clinical Text'].map(Remove_Discharge_Diagnosis))
print (df_notes)

def separate_diagnoses(diagnoses):
    pos0, pos1 = 0,0
    try:
        pos0 = diagnoses.index('discharge diagnoses') 
        pos1 = len("discharge diagnoses") + pos0
        cleaned_diagnoses = []
        indexes = set([j for j in range(pos0, pos1+1, 1)])    
        diagnoses  =  "".join([char for idx, char in enumerate(diagnoses) if idx not in indexes])
        diagnoses = diagnoses.split("\n")
        for word in diagnoses:
            if word=="":
                diagnoses.remove("")
        remove = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "", "."]
        for word in diagnoses:
            for r in remove:
                new_word = word.replace(r, "")
                word = new_word
            
            new_word = new_word[1:]
            cleaned_diagnoses.append(new_word)  
        return cleaned_diagnoses
    except:
        return None

df_notes['Discharge Diagnoses'] = df_notes['Discharge Diagnoses'].apply(lambda x : separate_diagnoses(x))

df_notes = df_notes.dropna()

print(df_notes['Discharge Diagnoses'])


#Join all diagnoses in a single list

diagnoses = df_notes['Discharge Diagnoses']
alldiagnoses = []
for diagnoselist in diagnoses:
    if diagnoselist != None:
        alldiagnoses += diagnoselist

distinct = []

for diagnose in diagnoses:
    if diagnose not in distinct:
        distinct.append(diagnose)

print("All diagnoses:", len(alldiagnoses))
print("Distinct diagnoses:", len(distinct))


def categorize_diagnoses(all_diagnoses):
    return None
