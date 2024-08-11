#!/usr/bin/env python
# coding: utf-8

# In[180]:


## This notebook is set up to run the program via the command line in the following format:
## $ python3 program.py training_data/training_data_file.xlsx output_folder
## It currently needs to be run in a directory that contains two subdirectories: "training_data" and "classifier_testing"
## The output directory needs to have a file called 'classifier_testing_parameters.xlsx' in order to work


# In[6]:





# In[ ]:


import sys

training_data = sys.argv[1]
output_subdirectory = sys.argv[2]


# In[16]:


#Generating ID no. for the test

import pandas as pd

import os

parameter_df = pd.read_excel(os.path.join(output_subdirectory, "classifier_testing_parameters.xlsx"))



test_id_list = parameter_df['test_ID'].to_list()

import random

test_ID = None

while True:
    rand_num = random.randint(1, 9999)
    formatted_rand_num = f"{rand_num:04d}"
    if int(formatted_rand_num) not in test_id_list:
        test_ID = int(formatted_rand_num)
        break

print(test_ID)



# In[20]:





# ## Setting test name and hyperparameters ##

# In[182]:


#Test hyperparameters
num_classes = 3
max_length = 512
bert_model_name = 'bert-base-uncased'
num_epochs = 5
learning_rate = 2e-5
batch_size = 16


# In[ ]:





# ## Imports ##

# In[183]:


import os


# In[184]:


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# In[185]:


from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


# In[186]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# ## Prepping training data ##

# In[188]:


original_df = pd.read_excel(training_data)
    
original_df.head()


# In[189]:


original_df.info()


# In[190]:


#Finding number of texts from which current training data is taken

a = original_df['filename'].nunique(dropna=True)
b = original_df['filename_x'].nunique(dropna=True)
c = original_df['filename_y'].nunique(dropna=True)
no_of_texts = a+b+c
print(no_of_texts)


# In[191]:


#Creating a cleaned up dataframe for training

index1 = []

for index, row in original_df.iterrows():
    values = [row['eric_category'], row['lucia_category'], row['sada_category']]
    common_label = None
    
    # Find the label that appears at least twice
    for value in set(values):
        if values.count(value) >= 2:
            common_label = value
            break
    
    # If a common label is found, save it and the 'text' column to the list
    if common_label:
        index1.append({'text': row['text'], 'category': common_label})

# Create a new dataframe from the list of rows
df = pd.DataFrame(index1)


# In[192]:


df.head()


# In[193]:


df.info()


# In[194]:


#Sanity check

df['category'].unique()


# In[195]:


#Getting number of training samples for each class

count_behavior = (df['category'] == 'behavior').sum()
count_mental = (df['category'] == 'mental').sum()
count_other = (df['category'] == 'other').sum()


# In[196]:


# Creating lists of training texts and numeric designations for classes

texts = df['text'].tolist()

designation_numeric = []

#df['designation']

for x in df['category']:
    if x == 'other':
        designation_numeric.append(0)
    elif x == 'mental':
        designation_numeric.append(1)
    elif x == 'behavior':
        designation_numeric.append(2)
    else:
        continue


# In[197]:


labels = torch.tensor(designation_numeric)
type(labels)


# In[200]:


#Sanity check
len(texts)


# In[199]:


len(labels)


# ## Setting up classes and functions for classifier ##

# In[169]:


#Creating a class (object and set of associate functions) to store the training data in a certain structure,
# and also query the training and output data. Class consists of the input texts, their integer labels, the BERT
# tokenizer used to prep the data for feeding to the classifier, and the max input length the model will take. 
#This class is a child class of the Pytorch "torch.utils.data.Dataset" parent/base class
#Sentences longer than the max input length will be truncated and the remainder discarded!

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


# In[170]:


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


# In[171]:


def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()


# In[172]:


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


# In[173]:


def predict_description(text, model, tokenizer, device, max_length):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    label_map = {0: "null", 1: "mental", 2: "behaviour"}
    return label_map[preds.item()]


# In[174]:


# Set path variables to save results. 

accuracy_file = str(test_ID) + "_accuracy_scores.docx"
classifier_name = "bert_classifier_" + str(test_ID) + ".pth"
accuracy_path = os.path.join(output_subdirectory, accuracy_file)
classifier_path = os.path.join(output_subdirectory, classifier_name)
parameters_file = os.path.join(output_subdirectory, "classifier_testing_parameters.xlsx")


# ## Assigning variables for classifier training ##

# In[175]:


train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)


# In[176]:


tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# ## Actually training the classifier ##

# In[177]:


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    if epoch == num_epochs - 1:
        final_accuracy = accuracy
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)
    
    ##UPDATE FILE NAME FOR EACH TEST!
    with open(accuracy_path, 'a') as f:
        f.write(report)
        
        

    


# ## Saving classifier and hyperparameters ##

# In[178]:


torch.save(model.state_dict(), classifier_path)


# In[21]:





# In[ ]:


try:
    df_existing = pd.read_excel(parameters_file)
except FileNotFoundError:
    # If the file doesn't exist, create an empty DataFrame with the same columns
    df_existing = pd.DataFrame(columns=['test_ID', 'model_file', 'score_file', 'training_size',
       'behavior_size', 'mental_size', 'other_size', 'base_model',
       'epochs', 'learning_rate', 'batch_size', 'max_length',
       'final_accuracy', 'training_diversity'])

# Define the new data to be added
training_size = len(labels)  

data = {
    "test_ID": [test_ID],
    "model_file": [classifier_name],
    "score_file": [accuracy_file],
    "training_size": [training_size],
    "behavior_size": [count_behavior],
    "mental_size": [count_mental],
    "other_size": [count_other],
    "base_model": [bert_model_name],
    "epochs": [num_epochs],
    "learning_rate": [learning_rate],
    "batch_size": [batch_size],
    "max_length": [max_length],
    "final_accuracy": [final_accuracy],
    "training_diversity": [no_of_texts]
}

# Create a new DataFrame with the new data
df_new = pd.DataFrame(data)

# Concatenate the existing DataFrame with the new DataFrame
df_updated = pd.concat([df_existing, df_new], ignore_index=True)

# Save the updated DataFrame back to the Excel file
df_updated.to_excel(parameters_file, index=False)

print(f"Data has been added to {parameters_file}")

