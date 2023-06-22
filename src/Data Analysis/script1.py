#setting the directories
import os
dir = "/home/ana.lopes/mscmulmiesclin_v0.1/data/mimiciii_onlynotes"
note_dir = os.listdir(dir)
notes_dir = []

for i in range(len(note_dir)):
    notes_dir.append(dir+"/"+note_dir[i])

#creating DataFrame with all clinical notes
import pandas as pd
notes_dict = {}
key = "note_"

for i in range(len(notes_dir)):
    file = open(notes_dir[i])
    note = file.read()
    file.close()
    notes_dict[key+str(i)] = note

df_notes = pd.DataFrame({"Note ID": notes_dict.keys(), "Clinical Text": notes_dict.values()})
print(df_notes.head())

#Preprocessing : token, pos, lemma
import stanza

def token(str):

    nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma')
    doc = nlp(str)
    tokens = []
    upos = []

    for sent in doc.sentences:
        for word in sent.words:
            tokens.append(word.lemma)
            upos.append(word.upos)
    return tokens, upos

def Remove_Discharge_Diagnosis(clinical_text):
    pos1, pos2 = 0,0
    clinical_text = clinical_text.lower()
    if "discharge diagnoses" in clinical_text:
        pos1 = clinical_text.index('discharge diagnoses') 
        for i in range(pos1, len(clinical_text)-1,1):
            if clinical_text[i] == "\n" and clinical_text[i+1]=="\n":
                pos2 = i+1
                break
    indexes = set([j for j in range(pos1, pos2+1, 1)])
    discharge_diagnoses = clinical_text[pos1:pos2]
    clinical_text  =  "".join([char for idx, char in enumerate(clinical_text) if idx not in indexes])
    return discharge_diagnoses, clinical_text



""""
df_notes_1 = df_notes.iloc[:1000]
df_notes_1['tokens'], df_notes_1['upos'] =zip(*df_notes_1['Clinical Text'].map(token))
df_notes_1.to_csv('1000_separeted.csv')




df_notes_2 = df_notes.iloc[1000:2000]
df_notes_2['tokens'], df_notes_2['upos'] =zip(*df_notes_2['Clinical Text'].map(token))
df_notes_2.to_csv('2000_separeted.csv')

df_notes_3 = df_notes.iloc[2000:3000]
df_notes_3['tokens'], df_notes_3['upos'] =zip(*df_notes_3['Clinical Text'].map(token))
df_notes_3.to_csv('3000.csv')

df_notes_4 = df_notes.iloc[3000:4000]
df_notes_4['tokens'], df_notes_4['upos'] =zip(*df_notes_4['Clinical Text'].map(token))
df_notes_4.to_csv('4000.csv')

df_notes_5 = df_notes.iloc[4000:5000]
df_notes_5['tokens'], df_notes_5['upos'] =zip(*df_notes_5['Clinical Text'].map(token))
df_notes_5.to_csv('5000.csv')
"""






