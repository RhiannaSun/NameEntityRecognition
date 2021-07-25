
import pandas as pd
import numpy as np
import re

from seqeval.metrics import f1_score, accuracy_score
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import Dataset,TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch
import emoji
from transformers import AutoModel, AutoTokenizer,BertForTokenClassification

import transformers
from transformers import BertForTokenClassification, AdamW

from preprocess import read_file,normalizeToken,clean_data

def tokenize_sentence(tokenized_texts, tokenizer):
    encoded_input= tokenizer(
        tokenized_texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True
        ,add_special_tokens=False
    )
    return encoded_input, tokenizer

def tokenize_tag(tags,tag2idx,encodings):
    labels = [[tag2idx[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        valid = len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] )
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels[:valid]
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

def train(train_data, dev_data, tag_values, tokenizer):
    # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

    tag2idx = {t: i for i, t in enumerate(tag_values)}
    idx2tag =  {value:key for key, value in tag2idx.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    model = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )

    model = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )

    epochs = 3
    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_data) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        model.cuda();
    print(f'Is CUDA available: {has_cuda}')
    print()

    loss_values, validation_loss_values = [], []
    batch_size=32
    for _ in range(epochs):
        # ========================================
        #               Training
        # ========================================
        model.train()
        total_loss = 0

        train_idx = np.arange(len(train_data))
        train_data_input = [x for (x,y) in train_data ]
        train_data_label = [y for (x,y) in train_data ]
        for i in range(0, len(train_data), batch_size):
            end = min(i+batch_size, len(train_data))
            tokenized_texts, labels = train_data_input[i:end], train_data_label[i:end]

            b_input, tokenizer= tokenize_sentence(tokenized_texts,tokenizer)
            b_input_mask = [x for x in b_input.attention_mask]
            b_input_ids = [x for x in b_input.input_ids]
            b_labels = tokenize_tag(labels,tag2idx, b_input)
            b_input_mask = torch.tensor(b_input_mask)
            b_input_ids = torch.tensor(b_input_ids)
            b_labels = torch.tensor(b_labels)

            if has_cuda:
                b_input_ids, b_input_mask, b_labels = b_input_ids.cuda(), b_input_mask.cuda(), b_labels.cuda()

            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_data)
        print("Average train loss: {}".format(avg_train_loss))

        loss_values.append(avg_train_loss)
      # ========================================
      #               Validation
      # ========================================

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        dev_data_input = [x for (x,y) in dev_data ]
        dev_data_label = [y for (x,y) in dev_data ]
        with torch.no_grad():
            for i in range(0, len(dev_data), batch_size):
                end = min(i+batch_size, len(dev_data))

                tokenized_texts, labels = dev_data_input[i:end], dev_data_label[i:end]
                b_input,tokenizer = tokenize_sentence(tokenized_texts,tokenizer)

                b_input_mask = [x for x in b_input.attention_mask]
                b_input_ids = [x for x in b_input.input_ids]
                b_labels = tokenize_tag(labels,tag2idx,b_input)
                b_input_mask = torch.tensor(b_input_mask)
                b_input_ids = torch.tensor(b_input_ids)
                b_labels = torch.tensor(b_labels)

                if has_cuda:
                    b_input_ids, b_input_mask, b_labels = b_input_ids.cuda(), b_input_mask.cuda(), b_labels.cuda()

                outputs = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels)

                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                eval_loss += outputs[0].mean().item()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)
                assert len(predictions) == len(true_labels)

        eval_loss = eval_loss / len(dev_data)
        validation_loss_values.append(eval_loss)
        print("Development loss: {}".format(eval_loss))

        pred_tags = [idx2tag[p_i] for p, l in zip(predictions, true_labels)
                                      for p_i, l_i in zip(p, l) if l_i != -100]
        valid_tags = [idx2tag[l_i] for l in true_labels
                                      for l_i in l if l_i != -100]
        print("Development Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Development F1-Score: {}".format(f1_score([pred_tags], [valid_tags])))
        print()

        return model

def predict(model, test_data, tag_values):

    tag2idx = {t: i for i, t in enumerate(tag_values)}
    idx2tag =  {value:key for key, value in tag2idx.items()}

    batch_size = 32
    predictions , offset = [], []
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            end = min(i+batch_size, len(test_data))

            tokenized_texts = test_data[i:end]
            b_input,tokenizer = tokenize_sentence(tokenized_texts,tokenizer)

            b_input_mask = [x for x in b_input.attention_mask]
            b_input_ids = [x for x in b_input.input_ids]
            b_input_offset = [x for x in b_input.offset_mapping]
            offset.extend(b_input_offset)
            b_input_mask = torch.tensor(b_input_mask)
            b_input_ids = torch.tensor(b_input_ids)

            if has_cuda:
                b_input_ids, b_input_mask = b_input_ids.cuda(), b_input_mask.cuda()

            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
            logits = outputs.logits.detach().cpu().numpy()
            # logits = outputs[1]
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    # get true predict output
    final = []
    for pred, off in zip(predictions, offset):
        pred = np.array(pred)
        off = np.array(off)
        true = pred[(off[:,0] == 0) & (off[:,1] != 0)]
        final.append(true)
    valid_tags = [[idx2tag[l_i] for l_i in l] for l in final]

    # write the output
    textfile = open("test_output.txt", "w")
    for element in valid_tags:
        for l in element:
            textfile. write(l + "\n")
        textfile.write("\n")
    textfile. close()

if __name__ == '__main__':
    train_path = '/content/drive/MyDrive/nlp_final/data/train/train.txt'
    dev_path = '/content/drive/MyDrive/nlp_final/data/dev/dev.txt'
    test_path = '/content/drive/MyDrive/nlp_final/data/test/test.nolabels.txt'
    train = read_file(train_path)
    train_tokenized_texts, train_labels = clean_data(train,tokenizer)
    dev = read_file(dev_path)
    dev_tokenized_texts, dev_labels = clean_data(dev,tokenizer)

    test = read_file(test_path, train=False)
    test_data =  clean_data(test,tokenizer, train=False)

    train_data = [(train_tokenized_texts[i], train_labels[i]) for i in range(len(train_labels))]
    dev_data = [(dev_tokenized_texts[i], dev_labels[i]) for i in range(len(dev_labels))]

    tag_values = list(set([x for y in train.tags.tolist() for x in y]))

    tag2idx = {t: i for i, t in enumerate(tag_values)}
    idx2tag =  {value:key for key, value in tag2idx.items()}

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    model = train(train_data,dev_data,tag_values, tokenizer)
    dev_data_input = [x for (x,y) in dev_data ]
    predict(model, dev_data_input, tag_values)
    predict(model, test_data,tag_values)