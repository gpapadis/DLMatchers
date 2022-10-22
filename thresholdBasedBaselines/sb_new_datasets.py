# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:50:07 2022

@author: G_A.Papadakis
"""

import numpy as np
import math
import pandas as pd
import re
import time

from nltk.stem import SnowballStemmer 
from nltk.corpus import stopwords

def cleanSentence(sentence):
    tokens = set(filter(None, re.split('[\\W_]', sentence.lower())))
    stemmed_words = set()
    for word in tokens:
        if word not in stop_words:
            stemmed_word = stemmer_ss.stem(word).strip()
            if stemmed_word:
                stemmed_words.add(stemmed_word)
    return stemmed_words

def get_similarities(clean, col_id1, col_id2, row):
    entity1 = str(row[col_id1])
    entity2 = str(row[col_id2])
    
    if clean:
        tokens1 = cleanSentence(entity1)
        tokens2 = cleanSentence(entity2)
    else:
        tokens1 = set(filter(None, re.split('[\\W_]', entity1.lower())))
        tokens2 = set(filter(None, re.split('[\\W_]', entity2.lower())))
        
    if tokens1 and tokens2:
        common_tokens = tokens1 & tokens2

        cs = len(common_tokens)/math.sqrt(len(tokens1) * len(tokens2))
        ds = 2 * len(common_tokens)/(len(tokens1) + len(tokens2))
        js = len(common_tokens)/(len(tokens1) + len(tokens2) - len(common_tokens))
        
        return cs, ds, js
    return 0, 0, 0

def get_similarity(clean, col_id1, col_id2, measure_id, row):
    entity1 = str(row[col_id1])
    entity2 = str(row[col_id2])

    if clean:
        tokens1 = cleanSentence(entity1)
        tokens2 = cleanSentence(entity2)
    else:
        tokens1 = set(filter(None, re.split('[\\W_]', entity1.lower())))
        tokens2 = set(filter(None, re.split('[\\W_]', entity2.lower())))

    if tokens1 and tokens2:
        common_tokens = tokens1 & tokens2

        if (measure_id == 0):
            return len(common_tokens)/math.sqrt(len(tokens1) * len(tokens2))
        elif (measure_id == 1):
            return 2 * len(common_tokens)/(len(tokens1) + len(tokens2))
        elif (measure_id == 2):    
            return len(common_tokens)/(len(tokens1) + len(tokens2) - len(common_tokens))
        
    return 0

stemmer_ss = SnowballStemmer("english")   
stop_words = set(stopwords.words('english'))

similarities = ['CS', 'DS', 'JS']
main_dir = '/data/newDatasets/'
datasets = ['Dn1', 'Dn2', 'Dn5', 'Dn6', 'Dn7']

for dataset in datasets:
    current_dir = main_dir + dataset
    print('\n\n' + current_dir)
    
    train = pd.read_csv(current_dir + '/train_set.csv', na_filter=False)
    valid = pd.read_csv(current_dir + '/valid_set.csv', na_filter=False)
    test = pd.read_csv(current_dir + '/test_set.csv', na_filter=False)
    
    d1_cols, d2_cols = [], []
    for index in range(4, len(train.columns)):
        if (train.columns[index].startswith("left")):
            d1_cols.append(index)
        elif (train.columns[index].startswith("right")):
            d2_cols.append(index)

    print('D1 Colums', d1_cols)
    print('D2 Colums', d2_cols)

    time_1 = time.time()

    best_attr_F1, best_attr_id,  best_clean_id, best_measure_id, best_threshold = -1, -1, -1, -1, -1
    for attr_index in range(len(d1_cols)):
        for clean_id in range(2):
            best_thresholds = []
            train['CS'], train['DS'], train['JS'] = zip(*train.apply(lambda row : get_similarities(clean_id, d1_cols[attr_index], d2_cols[attr_index], row), axis = 1))
            for measure_id in range(len(similarities)):
                best_F1, bestThr = -1, -1
                for threshold in np.arange(0.01, 1.00, 0.01):                
                    train['pred_label'] = threshold <= train[similarities[measure_id]]
                    
                    tp = len(train[(train['label'] == 1) & train['pred_label']])
                    fp = len(train[(train['label'] == 0) & train['pred_label']])
                    fn = len(train[(train['label'] == 1) & (train['pred_label'] == False)])
    
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0
                    if ((0 < precision) & (0 < recall)):
                        f1 = 2 * precision * recall / (precision + recall)
                        if (best_F1 < f1):
                            best_F1 = f1
                            bestThr = threshold
                print(best_F1, bestThr)
                best_thresholds.append(bestThr)
    
        for clean_id in range(2):
            valid['CS'], valid['DS'], valid['JS'] = zip(*valid.apply(lambda row : get_similarities(clean_id, d1_cols[attr_index], d2_cols[attr_index], row), axis = 1))
            for measure_id in range(len(similarities)):             
                valid['pred_label'] = best_thresholds[measure_id] <= valid[similarities[measure_id]]
                
                tp = len(valid[(valid['label'] == 1) & valid['pred_label']])
                fp = len(valid[(valid['label'] == 0) & valid['pred_label']])
                fn = len(valid[(valid['label'] == 1) & (valid['pred_label'] == False)])
    
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                if ((0 < precision) & (0 < recall)):
                    f1 = 2 * precision * recall / (precision + recall)
                    if (best_attr_F1 < f1):
                        best_attr_F1 = f1
                        best_attr_id = attr_index
                        best_clean_id = clean_id
                        best_measure_id = measure_id
                        best_threshold = best_thresholds[measure_id]
                    
    time_2 = time.time()
    
    test['sim'] = test.apply(lambda row : get_similarity(best_clean_id, d1_cols[best_attr_id], d2_cols[best_attr_id], best_measure_id, row), axis = 1)

    test['pred_label'] = best_threshold <= test['sim']
    tp = len(test[(test['label'] == 1) & test['pred_label']])
    fp = len(test[(test['label'] == 0) & test['pred_label']])
    fn = len(test[(test['label'] == 1) & (test['pred_label'] == False)])

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall)
    
    time_3 = time.time()
    
    print('Best threshold', best_threshold, similarities[best_measure_id])
    print('Best attribute', best_attr_F1, train.columns[d1_cols[best_attr_id]], train.columns[d2_cols[best_attr_id]])
    print('Final F1', 2 * precision * recall / (precision + recall))
    print('Training time (sec)', time_2-time_1)
    print('Testing time (sec)', time_3-time_2)