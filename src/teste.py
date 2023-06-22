import pandas as pd

train_dataset = pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/data/MP_IN_adm_train.csv")
val_dataset = pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/data/MP_IN_adm_val.csv")
test_dataset = pd.read_csv("/home/ana.lopes/mscmulmiesclin_v0.1/data/MP_IN_adm_test.csv")
print(train_dataset['text'][8])

