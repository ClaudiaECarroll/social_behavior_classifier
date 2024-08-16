#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Imports ##


# In[2]:


import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# In[ ]:


# Set variables to save results. 

training_subdirectory = "/home/claudiac/projects/social_behavior_classifier/training_data/"
output_subdirectory = "/home/claudiac/projects/social_behavior_classifier/classifier_testing/"




# ## Setting test name and hyperparameters ##

# In[3]:


#Setting number of classes for classifier. 
num_classes = 3


# In[4]:


#Importing hyperparameter spreadsheet

params = pd.read_excel("classifier_testing/hyperparameter_testing.xlsx")




# In[16]:


td = params['training_data'].dropna().tolist()
bm = params['bert_model_name'].dropna().tolist()
lr = params['learning_rate'].dropna().tolist()
ne = params['num_epochs'].dropna().tolist()
bs = params['batch_size'].dropna().tolist()
ml = params['max_length'].dropna().tolist()



strings_to_remove = ['roBERTa-base', 'roBERTa-large']

# Loop through the list and remove the strings if found
bm = [s for s in bm if s not in strings_to_remove]






# In[22]:





# In[15]:





# In[30]:


import itertools

combinations = list(itertools.product(td, bm, lr, ne, bs, ml))

# Convert to DataFrame
df_hyper = pd.DataFrame(combinations, columns=['td', 'bm', 'lr', 'ne', 'bs', 'ml'])

df_hyper_unique = df_hyper.drop_duplicates()




# In[31]:


#2880 is too many permutations!! Lets pick a random 100 for testing

sample_size = 100

num_columns = df_hyper_unique.shape[1]

samples_per_group = sample_size // num_columns

sampled_df = pd.DataFrame(columns=df_hyper_unique.columns)

for column in df_hyper_unique.columns:
    # Group by the unique elements in the column and sample from each group
    groups = df_hyper_unique.groupby(column)
    sampled_group = groups.apply(lambda x: x.sample(n=samples_per_group, replace=False)).reset_index(drop=True)
    sampled_df = pd.concat([sampled_df, sampled_group], ignore_index=True)

# If the sampled_df has more than 100 rows due to overlap, sample again to ensure only 100 rows
if sampled_df.shape[0] > sample_size:
    sampled_df = sampled_df.sample(n=sample_size, replace=False).reset_index(drop=True)




# In[32]:





# In[41]:


for index, row in sampled_df.iterrows():
    td = os.path.join(training_subdirectory, row['td'])
    training_data = pd.read_excel(td)
    bert_model_name = row['bm']
    learning_rate = row['lr']
    num_epochs = int(row['ne'])
    batch_size = int(row['bs'])
    max_length = int(row['ml'])


# In[ ]:


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

    accuracy_file = str(test_ID) + "_accuracy_scores.docx"
    classifier_name = "bert_classifier_" + str(test_ID) + ".pth"
    accuracy_path = os.path.join(output_subdirectory, accuracy_file)
    classifier_path = os.path.join(output_subdirectory, classifier_name)
    parameters_file = os.path.join(output_subdirectory, "classifier_testing_parameters.xlsx")


# ## Prepping training data ##

# In[42]:



    # In[190]:


    #Finding number of texts from which current training data is taken

    a = training_data['filename'].nunique(dropna=True)
    b = training_data['filename_x'].nunique(dropna=True)
    c = training_data['filename_y'].nunique(dropna=True)
    no_of_texts = a+b+c


    # In[191]:


    #Creating a cleaned up dataframe for training

    index1 = []

    for index, row in training_data.iterrows():
        # Check if 'sada_category' column exists
        if 'sada_category' in training_data.columns:
            # Use all three columns
            values = [row['eric_category'], row['lucia_category'], row['sada_category']]
        else:
            # Use only 'eric_category' and 'lucia_category'
            values = [row['eric_category'], row['lucia_category']]
        
        common_label = None
        
        # Find the label that appears at least twice (or once if only two values are checked)
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
        train(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        if epoch == num_epochs - 1:
            final_accuracy = accuracy
        
        
        ##UPDATE FILE NAME FOR EACH TEST!
        with open(accuracy_path, 'a') as f:
            f.write(report)
            
            

        


    # ## Saving classifier and hyperparameters ##

    # In[178]:


    #Enabling this cell if you want to save the weights file

    #torch.save(model.state_dict(), classifier_path)


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

    print(f"Data for {test_ID} has been added to {parameters_file}. Final accuracy: {final_accuracy}")

