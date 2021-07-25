import pandas as pd

def read_file(path, train=True):
    with open(path) as f:
        lines = f.read().splitlines()
    if train:
        words = []
        tags = []
        sentences = []
        sentences_tags = []
        for i in lines:
            if len(i) > 0 :
                split = i.split('\t')
                words.append(split[0])
                tags.append(split[1])
            else:
                sentences.append(words)
                sentences_tags.append(tags)
                words = []
                tags = []

        data = pd.DataFrame(columns=['tweet','tags'])
        data['tweet'] = sentences
        data['tags'] = sentences_tags
    else:
        words = []
        sentences = []
        for i in lines:
            if len(i) > 0 :
                words.append(i)
            else:
                sentences.append(words)
                words = []

        data = pd.DataFrame(columns=['tweet'])
        data['tweet'] = sentences
    return data

def normalizeToken(token):
    # token = emoji.demojize(token)
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif token.startswith("#"):
        if len(token) < 2:
            return token
        else:
            return token[1:]
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def clean_data(data,tokenizer, train=True):

    data['norm_tweet'] = data['tweet'].apply(lambda x : [normalizeToken(y) for y in x])

    norm_data = data.norm_tweet.tolist()
    if train == False:
      return norm_data
    tags = data.tags.tolist()
    return norm_data, tags