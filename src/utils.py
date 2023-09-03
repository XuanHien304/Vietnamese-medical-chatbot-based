import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import json
import torch
import random
import openai
nltk.download('punkt')

openai.api_key = 'OPENAI_API_KEY'

stemmer = PorterStemmer()
punctuation = ['?', '.', ',', '!', ':', '/']

def token(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = sorted(set([w.lower() for w in tokenized_sentence
                                     if w not in punctuation]))
    bag = np.zeros(len(all_words), dtype=np.float32)
    for (id, w) in enumerate(all_words):
        if w in tokenized_sentence:
            bag[id] = 1
    return bag

def get_label(path):
    with open(path, 'r') as json_data:
        contents = json.load(json_data)
    tags = []
    X = []
    for content in contents['intents']:
        tag = content['tag']
        for pattern in content['patterns']:
            X.append(pattern)
            tags.append(tag)

    tags_set = sorted(set(tags))
    return tags_set, contents

def problem_response(searcher, sentence):
    hits = searcher.search(sentence)
    if len(hits) == 0:
        return 'Mình chưa hiểu ý bạn lắm, bạn có thể cho mình xin thêm thông tin được không nhỉ'
    doc = searcher.doc(hits[0].docid)
    json_doc = json.loads(doc.raw())
    answer = json_doc['contents']
    return answer

def disease_response(model, tokenizer, sentence, rdrsegmenter, tags_set, contents):
    sentence = ' '.join(rdrsegmenter.tokenize(sentence)[0])
    token = tokenizer.encode_plus(sentence, truncation=True,
                                        add_special_tokens=True,
                                        max_length=30,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        return_token_type_ids=False,
                                        return_tensors='pt')

    X_mask = token['attention_mask']
    X = token['input_ids']

    with torch.inference_mode():
        output = model(X, X_mask)
    preds = torch.argmax(output, dim=1)

    tag = tags_set[preds.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][preds].item()
    if prob > 0.85:
        for content in contents['intents']:
            if tag == content['tag']:
                answer = random.choice(content['responses'])
    else:
        answer = 'Mình chưa hiểu ý bạn lắm, bạn có thể cho mình xin thêm thông tin được không nhỉ'            
    return answer, prob

def chatgpt_response(sentence):
    messages = [{'role': 'system', 'content': 'You are an expert doctor'}]
    messages.append([{'role': 'user', 'content': sentence}])
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
    try:
        answer = response['choices'][0]['messages']['content'].replace('\n', '<br>')
    except:
        answer = 'Mình chưa hiểu ý bạn lắm, bạn có thể cho mình xin thêm thông tin được không nhỉ'
    
    return answer