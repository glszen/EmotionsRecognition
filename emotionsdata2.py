"""
**********Emotions Data Analysis**********
**********@author: gulsen************
"""
""" *********************************************************************** """
""" 
Libraries
"""

import os
import re
import string
import json
import emoji
import numpy as np
import pandas as pd
from sklearn import metrics
from bs4 import BeautifulSoup
from transformers import RobertaModel, RobertaConfig, logging
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AutoTokenizer, BertModel, BertConfig, AutoModel, AdamW
import warnings

warnings.filterwarnings('ignore') #Hide warnings
logging.set_verbosity_info()

pd.set_option("display.max_columns", None) #Set the column maximum width

""" *********************************************************************** """
"""
Data Import
"""

data2_train=pd.read_csv(r'C:\Users\LENOVO\Desktop\data_science\emotionsdata2\train.tsv', sep='\t', header =None, names=['Text','Class','ID'])
data2_dev=pd.read_csv(r'C:\Users\LENOVO\Desktop\data_science\emotionsdata2\dev.tsv', sep='\t', header=None, names=['Text','Class','ID'])

""" *********************************************************************** """

"""
Data Preparation
"""

data2_train['List of Classes']=data2_train['Class'].apply(lambda x: x.split(',')) #For edit labels
data2_train['Len of Classes']=data2_train['List of Classes'].apply(lambda x: len(x))

data2_dev['List of Classes']=data2_dev['Class'].apply(lambda x: x.split(','))
data2_dev['Len of Classes']=data2_dev['List of Classes'].apply(lambda x: len(x))

with open (r'C:\Users\LENOVO\Desktop\data_science\emotionsdata2\ekman_mapping.json') as file:
    ekman_mapping=json.load(file)
    
emotion_file=open(r'C:\Users\LENOVO\Desktop\data_science\emotionsdata2\emotions.txt')
emotion_list=emotion_file.read()
emotion_list=emotion_list.split("\n")

def idx2class(idx_list): #Created a function that indexes and returns emotion variables in the dataset
    arr=[]
    for i in idx_list:
        arr.append(emotion_list[int(i)])
    return arr

data2_train['Emotions'] = data2_train['List of Classes'].apply(idx2class) #Emotions variables is defined 'List Of Classes' label according to the function
data2_dev['Emotions']=data2_dev['List of Classes'].apply(idx2class) 

def EmotionMapping(emotion_list): #Mapping operation for emotion list
    map_list=[]
    
    for i in emotion_list:
        if i in ekman_mapping['anger']:
            map_list.append('anger')
        if i in ekman_mapping['disgust']:
            map_list.append('disgust')
        if i in ekman_mapping['fear']:
            map_list.append('fear')
        if i in ekman_mapping['joy']:
            map_list.append('joy')
        if i in ekman_mapping['sadness']:
            map_list.append('sadness')
        if i in ekman_mapping['surprise']:
            map_list.append('surprise')
        if i =='neutral': #neutral variable is not exist in emotion_list so it is called directly
            map_list.append('neutral')
            
    return map_list 

data2_train['Mapped Emotions']=data2_train['Emotions'].apply(EmotionMapping) #'Mapped Emotions' label is defined for 'Emotions' variables according to function
data2_dev['Mapped Emotions']=data2_train['Emotions'].apply(EmotionMapping)

data2_train['anger']=np.zeros((len(data2_train),1)) #Define a field of zeros for each variable
data2_train['disgust']=np.zeros((len(data2_train),1))
data2_train['fear']=np.zeros((len(data2_train),1))
data2_train['joy']=np.zeros((len(data2_train),1))
data2_train['sadness']=np.zeros((len(data2_train),1))
data2_train['surprise']=np.zeros((len(data2_train),1))
data2_train['neutral']=np.zeros((len(data2_train),1))

data2_dev['anger']=np.zeros((len(data2_dev),1))
data2_dev['disgust']=np.zeros((len(data2_dev),1))
data2_dev['fear']=np.zeros((len(data2_dev),1))
data2_dev['joy']=np.zeros((len(data2_dev),1))
data2_dev['sadness']=np.zeros((len(data2_dev),1))
data2_dev['surprise']=np.zeros((len(data2_dev),1))
data2_dev['neutral']=np.zeros((len(data2_dev),1))

for i in ['anger', 'disgust', 'fear', 'joy', 'sadness','surprise','neutral']: #Specifies the presence or absence of variables
    data2_train[i]=data2_train['Mapped Emotions'].apply(lambda x:1 if i in x else 0)
    data2_dev[i]=data2_dev['Mapped Emotions'].apply(lambda x:1 if i in x else 0)
    
data2_train.drop(['Class', 'List of Classes','Len of Classes','Emotions','Mapped Emotions'], axis=1, inplace=True) #Some labels removed for analysis
data2_dev.drop(['Class', 'List of Classes','Len of Classes','Emotions','Mapped Emotions'], axis=1, inplace=True)

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", #Text edits
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                       "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am",
                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 
                       "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                       "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                       "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
                       "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                       "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
                       "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", 
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s':'america', 'e.g':'for example'}

punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 
                'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 
                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                'demonetisation': 'demonetization'}

def clean_text(text):
    text=emoji.demojize(text)
    text=re.sub(r'\:(.*?)\:','',text)
    text=str(text).lower()
    text=re.sub('\[.*?\]','',text)
    text=BeautifulSoup(text,'lxml').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text

def clean_contractions(text,mapping):
    specials=["'","'","'","'"]
    for s in specials:
        text=text.replace(s,"'")
    for word in mapping.keys():
        if ""+word+"" in text:
            text=text.replace(""+word+"", ""+mapping[word]+"")

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text=text.replace(p,mapping[p]) 
    
    for p in punct:
        text=text.replace(p, f'{p}')
        
        specials= {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
        for s in specials:
            text=text.replace(s,specials[s])
        return text
    
def correct_spelling(x,dic):
    for word in dic.keys():
        x=x.replace(word,dic[word])
        return x

def remove_space(text):
    text=text.strip()
    text=text.split()
    return "".join(text)

def text_preprocessing_pipeline(text):
    text=clean_text(text)
    text=clean_contractions(text,contraction_mapping)
    text=clean_special_chars()
    text=correct_spelling()    
    text=remove_space()
    return text

data2_train.reset_index(drop=True).to_csv("train.csv", index=False)
data2_dev.reset_index(drop=True).to_csv("val.csv", index=False)

data2_train=data2_train.reset_index(drop=True)
data2_dev=data2_dev.reset_index(drop=True)

""" *********************************************************************** """

"""
Data PROCESSİNG
"""

device='cuda' if torch.cuda.is_available() else 'cpu' #Determining the processor to use

MAX_LEN=200 #definitions for model building
TRAIN_BATCH_SIZE=64
VALID_BATCH_SIZE=64
EPOCHS=10
LEARNING_RATE=2e-5
tokenizer=AutoTokenizer.from_pretrained('roberta-base')

target_cols=[col for col in data2_train.columns if col not in ['Text', 'ID']]
target_cols 

class BERTDataset(Dataset): #a class for NLP
    def __init__(self, data, tokenizer,max_len):
        self.data=data
        self.max_len=max_len
        self.text=data.Text
        self.targets=data[target_cols].values
       
    def __len__ (self):
        return len(self.data)
    
    def __getitem__ (self,index):
        text=self.text[index]
        inputs=self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True, #the sequences will be encoded with the special tokens relative to their model
            max_length=self.max_len, #maximum length of the returned list. Will truncate by taking into account the special tokens
            padding='max_length',
            return_token_type_ids=True #Whether to return token type IDs
        )
        ids=inputs['input_ids']
        mask=inputs['attention_mask']
        token_type_ids=inputs['token_type_ids']
        return {
          'ids' : torch.tensor(ids, dtype=torch.long), #list of tokenized input ids
          'mask': torch.tensor(mask, dtype=torch.long), #list of indices specifying which tokens should be attended to by the model
          'token_type_ids' : torch.tensor(token_type_ids, dtype=torch.long),
                
          'targets' : torch.sensor(self.targets[index], dtype=torch.float)
                
            }
train_dataset=BERTDataset(data2_train, tokenizer, MAX_LEN)
valid_dataset=BERTDataset(data2_dev, tokenizer, MAX_LEN)

train_loader= DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True)
valid_loader= DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)

class MaskClassifier (RobertaPreTrainedModel): #Base class for all neural network modules
    def __init__ (self,config):
        super().__init__(config=config)
        self.roberta=RobertaModel(config)
        self.max_mask=10
        self.hidden_size=config.hidden_size
        self.linear1=torch.nn.Linear(2*self.hidden_size, self.hidden_size)
        self.linear2=torch.nn.Linear(self.hidden_size, self.max_mask + 1)
        self.softmax=torch.nn.Softmax(dim=1)
        
        self.init_weights()
        
MaskClassifier.from_pretrained(RobertaPreTrainedModel)
        
def forward(self, ids, mask,token_type_ids):
      _, features = self.roberta(ids,attension_mask=mask,token_type_ids=token_type_ids, return_dict=False)
       
      output=self.fc(features)
      return output
    
model = MaskClassifier.from_pretrained("roberta-base")

# MaskClassifier.from_pretrained(RobertaPreTrainedModel);

def loss_fn(outputs, targets): # a callable taking a prediction tensor
    return torch.nn.BCEWithLogitsLoss()(outputs, targets) #BCE loss is used for the binary classification tasks, BCEWithLogitsLoss:Binary cross-entropy with logits loss combines a Sigmoid layer and the BCELoss in one single class.

optimizer= AdamW(params=model.parameters(),lr=LEARNING_RATE, weight_decay=1e-6) 

def train(epoch):
    model.train()
    for _, data in enumerate(train_loader,0):
        ids=data['ids'].to(device, dtype=torch.long)
        mask=data['mask'].to(device, dtype=torch.long)
        token_type_ids=data['token_type_ids'].to(device, dtype=torch.long)
        targets=data['targets'].to(device,dtype=torch.float)
        
        outputs=model(ids,mask,token_type_ids)
        
        loss=loss_fn(outputs,targets)
        if _%500 == 0: 
            print(f'Epoch:{epoch}, Loss: {loss.item()}')
            
        loss.backward() #computes the partial derivative of the output f with respect to each of the input variables
        optimizer.step() #to zero the gradient
        optimizer.zero_grad()
        
    for epoch in range(EPOCHS):
        train(epoch)
        
def validation():  #helps to improve the security of code
    model.eval() #Sets model in evaluation (inference) mode
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad:  #no_grad:Context-manager that disabled gradient calculation.
        for _, data in enumerate(valid_loader,0):
            ids=data['ids'].to(device,dtype=torch.long)
            mask=data['mask'].to(device, dtype=torch.long)
            token_type_ids=data['token_type_ids'].to(device, dtype=torch.long)
            targets=data['targets'].to(device,dtype=torch.float)
            outputs=model(ids,mask,token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
    
    output, targets=validation()
    outputs=np.array(outputs) >= 0.5
    accuracy= metrics.accuracy_score(targets, outputs)
    f1_score_micro= metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro= metrics.f1_score(targets,outputs, average='macro')
    print(f"Accuracy Score={accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro)={f1_score_macro}")
    
    # torch.save(model.state_dict(), 'model.bin') #Saving and loading models
    
    model.save_pretrained("here")
    MaskClassifier.from_pretrained("here")
        