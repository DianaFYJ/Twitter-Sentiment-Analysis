import random
import csv
from os import mkdir

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch import nn
from transformers import Trainer

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_cosine_schedule_with_warmup, BertTokenizer
from transformers import BertModel
from transformers import TrainingArguments

from tqdm.auto import tqdm, trange
import pandas as pd
tqdm.pandas()

from datasets import Dataset

from sklearn.metrics import f1_score

from preprocessing import *

SEED = 517


'''define functions for tokenizing'''
def tokenize_text(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=64)

'''succeed Trainer'''
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        outputs = model(**inputs)
        # print(inputs)
        logits = outputs.get('logits')
        labels = inputs.get('labels')
        loss_func = nn.CrossEntropyLoss(weight = class_weight).to(device)
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    return {'f1':f1}

if __name__ == "__main__":

    '''set device to GPU or CPU'''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    '''save data to the defined path'''
    # path = '/content/drive/My Drive/ml_bert/'
    path = '../'

    ''' import dataset, preprocess and save in a tsv file '''
    with open(path+'data/train_pos_full.txt', 'r', encoding='utf-8') as pos,\
            open(path+'data/train_neg_full.txt', 'r', encoding='utf-8') as neg,\
            open(path+'data/train_clean.tsv', 'w', encoding='utf-8') as out:
        print('label\ttweet', file=out)
        for l in tqdm(neg, total=1250000, desc='Neg'):
            print('0\t' + preprocess(l), file=out)
        for l in tqdm(pos, total=1250000, desc='Pos'):
            print('1\t' + preprocess(l), file=out)

    '''load tsv file'''
    train_df = pd.read_csv(path+'data/train_clean.tsv', delimiter='\t', index_col=False)

    '''drop duplicate and null rows'''
    train_df = train_df.drop_duplicates()
    train_df = train_df.dropna()

    '''Define tokenizer'''
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    '''Drop the automatically generated items'''
    train_dataset = Dataset.from_pandas(train_df).remove_columns('__index_level_0__').rename_column('tweet','text')

    '''Tokenize train data'''
    train_dataset = train_dataset.map(tokenize_text,batched=True)

    '''to save train_dataset'''
    torch.save(train_dataset, path+'data/train_dataset.pt')

    '''split into train and val'''
    train_len = int(0.95*len(train_dataset))
    val_len = len(train_dataset) - train_len
    train,val = torch.utils.data.random_split(train_dataset, [train_len,val_len])

    '''load test data'''
    with open(path+'data/test_data.txt', 'r', encoding='utf-8') as test_file:
        lst = [line.rstrip('\n').split(',', 1) for line in test_file]
        test_df = pd.DataFrame(lst, columns=['id', 'tweet'])

    '''clean'''
    test_data = test_df['tweet'].apply(preprocess).to_frame()

    '''tokenize'''
    test_data = Dataset.from_pandas(test_data).rename_column('tweet','text').map(tokenize_text)

    '''Set Parameters for Trainer'''

    batch_size = 64
    logging_steps = len(train)
    output_dr = path + 'bert/'
    class_weight = torch.tensor([0.5,0.5])
    LR = 3e-5
    MAX_GRAD_NORM = 1

    training_args = TrainingArguments(output_dir=output_dr, num_train_epochs=10, learning_rate=LR, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, weight_decay=0.01, evaluation_strategy='steps', eval_steps=20000, logging_steps=logging_steps, save_steps=20000, push_to_hub=False, load_best_model_at_end=True)

    '''Set from the pre-trained model'''
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2).to(device)

    '''It is also practical to load the model from the checkpoint in case of a sudden disconnect or crash'''
    # model = AutoModelForSequenceClassification.from_pretrained(path+'checkpoint-120000-1219', num_labels = 2)

    trainer = WeightedLossTrainer(model=model, args=training_args, compute_metrics=compute_metrics, train_dataset=train, eval_dataset=val)

    trainer.train()

    '''use the finished model and trainer above, or load model from certain checkpoint as follows''' 
    # model = AutoModelForSequenceClassification.from_pretrained(path+'checkpoint-50000', num_labels = 2)
    # trainer = WeightedLossTrainer(model = model, args=training_args, train_dataset=train, eval_dataset=val)

    '''predict results'''
    preds = trainer.predict(test_data).predictions

    '''get labels'''
    test_label = []
    for i in range(len(preds)):
        if preds[i][0]>preds[i][1]:
            test_label.append(-1)
        else:
            test_label.append(1)

    '''export results'''
    test_id = test_df['id'].values.tolist()
    with open(path+'submission_bert.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(test_id, test_label):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})