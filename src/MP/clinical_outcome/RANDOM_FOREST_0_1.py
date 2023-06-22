
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import ast
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score


import itertools

#upload csv to dataframe
test_dataset = pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/prob_labels_mp.csv",  index_col=[0])

#convert strings in lists
#test_dataset['code_predictions'] = test_dataset['code_predictions'].apply(lambda x: ast.literal_eval(x))
test_dataset['code_probabilities'] = test_dataset['code_probabilities'].apply(lambda x: ast.literal_eval(x))


#function that deletes words and 4 digit codes from predicted labels
def delete_4_digit_codes(codes):
    cleaned_codes = []
    for code in codes:
        if code[0].isalpha() and len(code)>4:
           continue
        elif not code[0].isalpha() and len(code)>3:
            continue
        elif len(code) == 1:
            continue
        elif code[0].isalpha() and code[1].isalpha():
            continue
        cleaned_codes.append(code)
    return cleaned_codes

#upload model
tokenizer = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
labels = model.config.id2label

#remove 4 digit codes and words from labels dictionary
labels = dict(itertools.islice(labels.items(), 5543))
new_labels = {}
i = 0 
indexes = []
for idx, code in labels.items():
    if code[0].isalpha() and len(code)>4:
        continue
    elif not code[0].isalpha() and len(code)>3:
        continue
    new_labels[i] = code
    indexes.append(int(idx))
    i+=1

def remove_probabilities(indexes, prob_vec):
    new_prob_vec = []
    for i in range(len(prob_vec)):
        if i in indexes:
            new_prob_vec.append(prob_vec[i])
    return new_prob_vec

def predicted_codes(prob_vec, labels, t):
    icd_codes = []
    for i in range(len(prob_vec)):
        if prob_vec[i]>=t:
            icd_codes.append(labels[i])
    return icd_codes

#create binary vectores for metrics
def create_binary_vec (codes, labels):
    prob_vec = [0]*len(labels.keys())
    for i in range(len(labels.keys())):
        if labels[i] in codes:
            prob_vec[i] = 1
    return prob_vec

def delete_probs(prob_vec, t):
    new_prob_vec = []
    for val in prob_vec:
        if val>= t:
            new_prob_vec.append(val)
    return new_prob_vec

import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

test_dataset['code_probabilities'] = test_dataset['code_probabilities'].apply(lambda x : remove_probabilities(indexes,x))
print('done1')
test_dataset['code_predictions'] = test_dataset['code_probabilities'].apply(lambda x : predicted_codes(x, new_labels, 0.5))
print('done2')
#test_dataset['code_probabilities'] = test_dataset['code_probabilities'].apply(lambda x : delete_probs(x, 0.5))
print('done3')

test_dataset['y_pred'] = test_dataset['code_predictions'].apply(lambda x : create_binary_vec(x, new_labels))

#creating features

codes = list(test_dataset["y_pred"])


def Extract(lst,i):
    return [item[i] for item in lst]

def create_features(list):
    i = 0
    features = []
    while i <1248:
        f = Extract(list,i)
        features.append(f)
        i=i+1
    return features

features_vec = create_features(codes)
#print(features)

#insert features into dataframe
column_names = list(new_labels.values())
  
features= pd.DataFrame(codes, columns=column_names)
print(features)


labels = test_dataset['hospital_expire_flag'].to_numpy()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)

tree =  RandomForestClassifier() 
tree.fit(X_train, y_train) 
pred_test = tree.predict(X_test)
pred_train =tree.predict(X_train)

print("Accuracy in test dataset:" , accuracy_score(pred_test, y_test))
print("Balanced Accuracy in test dataset:" , balanced_accuracy_score(pred_test, y_test))
print("Precision in test dataset:" , precision_score(pred_test, y_test))
print("Roc AUC in test dataset:" , roc_auc_score(pred_test, y_test))
print(classification_report(pred_test, y_test))