# CS5740 final

In this project, we built a named-entity recognizer for Twitter text. 
Given a tweet, the tokenizer can identify sub-spans of words that represent named entities: B (beginning of the named entity), I (continuation of a named entity), or O (token that is not part of a named entity). We trained the model with annotated raw tweets data as describe in Section \ref{sec:data}. \\
We built the tokenizer on the BERT (Bidirectional Encoder Representations from Transformers), the state-of-art pre-trained language representation model proved to give faster and better performance on language task.
We experiment on three pretrained model: BERT base, BERT large and RoBERTa. F1 score is used to evaluate the performance of model. 





## Execution Instructions

To setup the environment for this project, use Python 3.8.x and install all requirements.
```
pip install -r requirements.txt
```

Run the following command to train the model. This will train the data on 500,000 sentences.
```
python model.py
```

This will generate an `embeddings.txt` which contains word embeddings for words occuring in the development and test set. Their cosine similarity and Spearman's rank can be computed as follows:
```
python tageval.py python tageval.py dev.txt dev.pred   
```

## Results

The results reported on the development and test set from these experiments are:

| Model | BERT base | BERT large | RoBERTa base |
| ----------- | ----------- | ---- | ----- |
| Development Set | 0.654 | 0.710 | 0.688|
| Test Set | 0.63221 | N/A | N/A |




Requires Python 2.x to run.
