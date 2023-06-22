import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

test_dataset = pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/data/MP_IN_adm_test.csv")
       
print(test_dataset.shape)


def truncate_clinical_note(txt):
    new_txt = ""
    txt_list = txt.split()
    txt_list_truncated = txt_list[0:312]
    print(txt_list_truncated)
    for word in txt_list_truncated:
        new_txt = new_txt + " " + word
    return new_txt

txt = test_dataset['text'][14]


def icd_codes(model, txt):

    input = txt
    tokenized_input = tokenizer(input, return_tensors="pt",max_length=512, padding=True, truncation=True)

    output = model(**tokenized_input)
    predictions = torch.sigmoid(output.logits)
    predicted_labels = [model.config.id2label[_id] for _id in (predictions > 0.3).nonzero()[:, 1].tolist()]
    return predicted_labels

def probabilities_codes(model, txt):

    input = txt
    tokenized_input = tokenizer(input, return_tensors="pt", max_length=512, padding=True, truncation=True)
  
    output = model(**tokenized_input)
    predictions = torch.sigmoid(output.logits)
    probabilities = predictions[0].tolist()[:5543]
    return probabilities


from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")

text_entries = np.array(test_dataset["text"])

code_probabilities = []
"""
code_predictions = []

for entry in tqdm(text_entries):
    code_predictions.append(icd_codes(model, entry))

code_predictions = pd.Series(code_predictions)
test_dataset['code_predictions'] = code_predictions.values
print('task 1 done')
"""
  
for entry in tqdm(text_entries):
    code_probabilities.append(probabilities_codes(model, entry))

code_probabilities = pd.Series(code_probabilities)
test_dataset['code_probabilities'] = code_probabilities.values

#test_dataset['predicted labels'] = test_dataset['text'].apply(lambda x: icd_codes(model, x))


#test_dataset['probabilities'] = test_dataset['text'].apply(lambda x: probabilities_codes(model, x))

test_dataset.to_csv('prob_labels_mp.csv')
print('done')

