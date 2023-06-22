from torch.utils import data
import random
from transformers import set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM, BioGptModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import itertools
import stanza
from matplotlib import pyplot as plt
import os
import string
from nltk.corpus import stopwords
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
from matplotlib import pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)




def remove_numbers(tokens):
    for word in tokens:
        try:
            int(str(word))
            tokens.remove(word)
        except ValueError:
            continue
    return tokens

    
def remove_stop_words(tokens_l):

    stop_words = set(stopwords.words('english'))
    cleaned_list = [word for word in tokens_l if (word not in stop_words and word!=None)]
    return cleaned_list

def lemma(str, engine):

    doc = engine(str)
    tokens = []
    upos = []

    for sent in doc.sentences:
        for word in sent.words:
            tokens.append(word.lemma)
            upos.append(word.upos)
    return tokens, upos

def remove_punctuation(token_list):
    punc = list(string.punctuation)
    cleaned_list = [word for word in token_list if word not in punc]
    return cleaned_list

def list_back_to_string(list_tokens):

    txt = " ".join(list_tokens)
    return txt
    
def single_processing (txt, engine):

    tokens, upos = lemma(txt, engine)
    tokens_no_stop_words = remove_stop_words(tokens)
    tokens_no_numbers = remove_numbers(tokens_no_stop_words)
    tokens_no_punc = remove_punctuation(tokens_no_numbers)
    txt = list_back_to_string(tokens_no_punc)
    return txt

def preprocessing(data,tokenizer,max_length, engine):
     
     engine = nlp
     cleaned_data = []
     i=0
     for txt in data:
        cleaned_txt = single_processing(txt, engine)
        cleaned_data.append(cleaned_txt)
        i=i+1
        print(i)

     encoded_inputs = tokenizer.batch_encode_plus(max_length= max_length,
                                                   truncation=True,
                                                     padding = True,
                                                       batch_text_or_text_pairs = cleaned_data,
                                                       return_tensors="pt")
     
     inputs = encoded_inputs['input_ids']
     masks = encoded_inputs['attention_mask']

     return inputs, masks


def create_dataloaders(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    tokenizer,
    max_length,
    engine,
    device,
    batch_size=16,
):

    print("Tokenizing data...")
    train_inputs, train_masks = preprocessing(
        x_train, 
        tokenizer, 
        max_length, 
        engine,  
    )

    val_inputs, val_masks = preprocessing(
        x_val, 
        tokenizer, 
        max_length, 
        engine
    )

    test_inputs, test_masks = preprocessing(
        x_test, 
        tokenizer, 
        max_length, 
        engine
    )

    

    train_labels = torch.stack([torch.tensor(label) for label in y_train])
    val_labels = torch.stack([torch.tensor(label) for label in y_val])
    test_labels = torch.stack([torch.tensor(label) for label in y_test])

    train_dataset = BioICD9(train_inputs,  train_masks, train_labels)
    val_dataset = BioICD9(val_inputs, val_masks, val_labels)
    test_dataset = BioICD9(test_inputs,  test_masks, test_labels)

    # Create the dataloaders
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=False,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=False,
    )

    final_train_dataloader = DeviceDataLoader(train_dataloader, device)
    final_val_dataloader = DeviceDataLoader(val_dataloader, device)
    final_test_dataloader = DeviceDataLoader(test_dataloader, device)

    return final_train_dataloader, final_val_dataloader, final_test_dataloader


class BioICD9(data.Dataset):
    def __init__(self, data, masks, labels):
        self.data = data
        self.masks = masks
        self.labels = labels.float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.masks[index], self.labels[index]

class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        batches = iter(self.dl)

        for b in batches:  # self.dl

            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    


def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list, tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device)
    
def create_binary_vec (codes, labels):

    prob_vec = [0]*len(labels.keys())
    for i in range(len(labels.keys())):
        if labels[i] in codes:
            prob_vec[i] = 1
    return np.array(prob_vec)


def df_column_to_array(df):

    return np.array(df["text"]), np.array(df["y_true"])



def initialize_model(biogpt_classifier, train_dataloader, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler."""

    # Tell PyTorch to run the model on GPU
    biogpt_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(
        biogpt_classifier.parameters(),
        lr=5e-6,  # Default learning rate
        # eps=1e-8,  # Default epsilon value
    )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5000,  # Default value
        num_training_steps=total_steps,
    )

    return biogpt_classifier, optimizer, scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader=None,
    epochs=4,
    evaluation=False,
    learning_curve=False,
    early_stopping=False,
    filename=None,
    save_path=None,
):
    """Train the BIOGPT model."""
    # Start training loop
    print("Start training...\n")

    val_losses = {}
    val_accs = {}
    train_losses = {}
    train_accs = {}
    criterion = nn.BCEWithLogitsLoss()

    if early_stopping:
        best_loss = float("inf")
        counter = 0
        patience = 5  # Set the patience value

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        batch_counts, batch_loss, total_loss = 0, 0, 0

        # For each batch of training data...
        for step, (b_input_ids, b_attn_mask, b_labels) in enumerate(train_dataloader):
            batch_counts += 1
            inputs = {
                "input_ids": b_input_ids,
                #"token_type_ids": b_token_ids,
                "attention_mask": b_attn_mask,
            }  # TODO: FIX THIS SPAGGETHI
            

            # Zero out any previously calculated gradients
            optimizer.zero_grad()
            outputs = model(**inputs)

            loss = criterion(outputs, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}",
                )

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}",
            )
            print("-" * 70)

            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                    # Save model checkpoint if needed
                else:
                    counter += 1

                if counter >= patience:
                    print("Early stopping triggered. Stopping training.")
                    break

            if learning_curve:
                tr_loss, tr_accuracy = evaluate(model, train_dataloader)
                train_losses[epoch_i + 1] = tr_loss
                train_accs[epoch_i + 1] = tr_accuracy
                val_losses[epoch_i + 1] = val_loss
                val_accs[epoch_i + 1] = val_accuracy

        print("\n")

    if save_path is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(save_path, f"bertimbau_{epochs}_epochs.pth"),
        )

        if learning_curve:
            plot_learning_curves(epochs, train_losses, train_accs, val_losses, val_accs, filename, save_path)

    print("Training complete!")

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    crit = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        model.eval()
        # Tracking variables
        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for b_input_ids, b_attn_mask, b_labels in val_dataloader:

            inputs = {"input_ids": b_input_ids, "attention_mask": b_attn_mask}

            # Compute logits
            logits = model(**inputs)
            loss = crit(logits, b_labels)

            val_loss.append(loss.item())

            # Get the predictions
            # preds = torch.argmax(logits, dim=1).flatten()
            _, preds = torch.max(logits, 1)

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        model.train()

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy




def plot_learning_curves(n_epochs, train_losses, train_accs, val_losses, val_accs, filename, save_path):
    # Retrieve each dictionary's values
    train_ls = train_losses.values()
    val_ls = val_losses.values()
    train_acs = train_accs.values()
    val_acs = val_accs.values()

    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, n_epochs + 1)

    # PLOT LOSS LEARNING CURVE
    fig = plt.figure()
    # Plot and label the training and validation loss values
    plt.plot(epochs, train_ls, label="Training Loss")
    plt.plot(epochs, val_ls, label="Validation Loss")

    # Add in a title and axes labels
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Set the tick locations
    plt.xticks(np.arange(0, n_epochs + 1, 10))

    # Display the plot
    plt.legend(loc="best")
    fig.savefig(os.path.join(save_path, f"{filename}_loss_learning_curve.svg"))

    # PLOT ACCURACY LEARNING CURVE
    fig2 = plt.figure()
    # Plot and label the training and validation accuracy values
    plt.plot(epochs, train_acs, label="Training Accuracy")
    plt.plot(epochs, val_acs, label="Validation Accuracy")

    # Add in a title and axes labels
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    # Set the tick locations
    plt.xticks(np.arange(0, n_epochs + 1, 10))

    # Display the plot
    plt.legend(loc="best")
    fig2.savefig(os.path.join(save_path, f"{filename}_accuracy_learning_curve.svg"))

def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """

    all_logits = []
    all_labels = []
    with torch.no_grad():
        model.eval()

        # For each batch in our test set...
        for b_input_ids, b_token_ids, b_attn_mask, labels in test_dataloader:
            inputs = {"input_ids": b_input_ids, "token_type_ids": b_token_ids, "attention_mask": b_attn_mask}

            # Compute logits
            logits = model(**inputs)["logits"]
            all_logits.append(logits)
            all_labels.append([label for label in labels])

        final_labels = torch.stack([target for targets in all_labels for target in targets]).cpu().numpy()

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = func.softmax(all_logits, dim=1).cpu().numpy()

        # Get predictions from the probabilities
        preds = np.argmax(probs, axis=1).flatten()

        model.train()

    return probs, preds, final_labels

def bert_evaluation(y_true, y_pred, y_pred_proba, binary=True):
    """Computes evaluation metrics for the predictions attained.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        y_pred_proba (np.array): Predicted probabilities.

    Returns:
        results (dict): Evaluation metrics.
    """
    if binary:
        roc = roc_auc_score(y_true, y_pred_proba, average="macro")
        ap = average_precision_score(y_true, y_pred_proba, average="macro")
    else:
        roc = roc_auc_score(y_true, y_pred_proba, average="macro", multi_class="ovo")
        ap = 0.0

    return {
        "ba": [balanced_accuracy_score(y_true, y_pred)],
        "f1": [f1_score(y_true, y_pred, average="macro")],
        "roc_auc": [roc],
        "average_precision": [ap],
        "recall": [recall_score(y_true, y_pred, average="macro")],
        "precision": [precision_score(y_true, y_pred, average="macro")],
        "cm": [confusion_matrix(y_true, y_pred)],
    }

def get_device():

 

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))

 

    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

 

    return device

class CustomModel(nn.Module): 
    def __init__(self, base_model, extra_layer): 
        super(CustomModel, self).__init__()
        self.base_model = base_model 
        self.extra_layer = extra_layer 
        
    def forward(self, input_ids, attention_mask): 
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask) 
        logits = self.extra_layer(outputs.last_hidden_state[:, 0, :]) 
        
        return logits

model = BioGptModel.from_pretrained("microsoft/biogpt", num_labels=1248)
lin = torch.nn.Linear(model.config.hidden_size, 1248)
model = CustomModel(model, lin)

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
set_seed(42)


model1 = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
labels = model1.config.id2label
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma', use_gpu=False)


#remove 4 digit icd-9 codes and words from labels dictionary
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

#model.output_projection.out_features = 1268
train_dataset = pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/data/MP_IN_adm_train.csv")
val_dataset = pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/data/MP_IN_adm_val.csv")
test_dataset = pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/data/MP_IN_adm_test.csv")


train_dataset['y_true'] = train_dataset['short_codes'].apply(lambda x : create_binary_vec(x, new_labels))
val_dataset['y_true'] = val_dataset['short_codes'].apply(lambda x : create_binary_vec(x, new_labels))
test_dataset['y_true'] = val_dataset['short_codes'].apply(lambda x : create_binary_vec(x, new_labels))

train_dataset = train_dataset[:2]
val_dataset = val_dataset[:2]
test_dataset = test_dataset[:2]

x_train , y_train = df_column_to_array(train_dataset)
x_val, y_val  = df_column_to_array(val_dataset)
x_test, y_test = df_column_to_array(test_dataset)

device = 'cpu'
batch_size = 1

final_train_dataloader, final_val_dataloader, final_test_dataloader = create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, tokenizer, max_length=512, engine=nlp, device=device, batch_size=batch_size)

biogpt_classifier, optimizer, scheduler = initialize_model(model, final_train_dataloader, epochs=4)

train(
    biogpt_classifier,
    optimizer,
    scheduler,
    final_train_dataloader,
    val_dataloader=final_val_dataloader,
    epochs=5,
    evaluation=True,
    learning_curve=True,
    early_stopping=False,
    filename="biogpt",
    save_path="/home/ana.lopes/mscmulmiesclin_v0.1/data/results"
)









