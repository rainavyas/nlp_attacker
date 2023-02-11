import random

from tqdm import tqdm 
from copy import deepcopy
from typing import List, Dict, Tuple
from datasets import load_dataset

def load_data(data_name:str, cache_dir:str=None, lim:int=None)->Tuple['train', 'val', 'test']:
    data_ret = {
        'imdb'    : _load_imdb,
        'twitter' : _load_twitter,
        'dbpedia' : _load_dbpedia,
        'rt'      : _load_rotten_tomatoes,
        'sst'     : _load_sst,
        'yelp'    : _load_yelp,
        'sst-glue': _load_sst_glue
    }
    return data_ret[data_name](cache_dir, lim)

def _load_imdb(cache_dir, lim:int=None)->List[Dict['text', 'label']]:
    dataset = load_dataset("imdb", cache_dir=cache_dir)
    train_data = list(dataset['train'])[:lim]
    train, val = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])[:lim]
    return train, val, test

def _load_twitter(cache_dir, lim:int=None)->List[Dict['text', 'label']]:
    # Source: https://www.kaggle.com/datasets/parulpandey/emotion-dataset?select=test.csv
    base_path = f'{cache_dir}/twitter/'
    CLASS_TO_IND = {
        '2': 2, # love
        '1': 1, # joy
        '4': 4, # fear
        '3': 3, # anger
        '5': 5, # surprise
        '0': 0, # sadness
    }
    train = _read_file(f'{base_path}training.csv', CLASS_TO_IND)
    val = _read_file(f'{base_path}validation.csv', CLASS_TO_IND)
    test = _read_file(f'{base_path}test.csv', CLASS_TO_IND)
    return train, val, test

def _load_dbpedia(cache_dir, lim:int=None):
    dataset = load_dataset("dbpedia_14", cache_dir=cache_dir)
    print('loading dbpedia- hang tight')
    train_data = dataset['train'][:lim]
    train_data = [_key_to_text(ex) for ex in tqdm(train_data)]
    train, val = _create_splits(train_data, 0.8)
        
    test  = dataset['test'][:lim]
    test = [_key_to_text(ex) for ex in test]
    return train, val, test

def _load_rotten_tomatoes(cache_dir, lim:int=None):
    dataset = load_dataset("rotten_tomatoes", cache_dir=cache_dir)
    train = list(dataset['train'])[:lim]
    val   = list(dataset['validation'])[:lim]
    test  = list(dataset['test'])[:lim]
    return train, val, test

def _load_yelp(cache_dir, lim:int=None):
    dataset = load_dataset("yelp_polarity", cache_dir=cache_dir)
    train_data = list(dataset['train'])[:lim]
    train, val = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])[:lim]
    return train, val, test
    
def _load_sst(cache_dir, lim:int=None)->List[Dict['text', 'label']]:
    dataset = load_dataset("gpt3mix/sst2", cache_dir=cache_dir)
    train = list(dataset['train'])[:lim]
    val   = list(dataset['validation'])[:lim]
    test  = list(dataset['test'])[:lim]
    
    train = [_invert_labels(ex) for ex in train]
    val   = [_invert_labels(ex) for ex in val]
    test  = [_invert_labels(ex) for ex in test]
    return train, val, test

def _load_sst_glue(cache_dir, lim:int=None)->List[Dict['text', 'label']]:
    dataset = load_dataset('glue', 'sst2', cache_dir=cache_dir)
    train_data = list(dataset['train'])
    train, val = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])
    
    train, val, test = _rename_keys(train, val, test, old_key='sentence', new_key='text')
    return train, val, test

   

def _read_file(filepath, CLASS_TO_IND):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]

    examples = []
    for line in lines:
        items = line.split(',')
        try:
            examples.append({'text':items[0], 'label':CLASS_TO_IND[items[1]]})
        except:
            pass
    return examples

def _create_splits(examples:list, ratio=0.8)->Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio*len(examples))
    
    random.seed(1)
    random.shuffle(examples)
    
    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2

def _key_to_text(ex:dict, old_key='content', new_key='text'):
    """ convert key name from the old_key to 'text' """
    ex = ex.copy()
    ex[new_key] = ex.pop(old_key)
    return ex

def _rename_keys(train:list, dev:list, test:list, old_key:str, new_key:str):
    train = [_key_to_text(ex, old_key, new_key) for ex in train]
    dev   = [_key_to_text(ex, old_key, new_key) for ex in dev]
    test  = [_key_to_text(ex, old_key, new_key) for ex in test]
    return train, dev, test

def _multi_key_to_text(ex:dict, key1:str, key2:str):
    """concatenate contents of key1 and key2 and map to name text"""
    ex = ex.copy()
    ex['text'] = ex.pop(key1) + ' ' + ex.pop(key2)
    return ex

def _invert_labels(ex:dict):
    ex = ex.copy()
    ex['label'] = 1 - ex['label']
    return ex

def _map_labels(ex:dict, map_dict={-1:0, 1:1}):
    ex = ex.copy()
    ex['label'] = map_dict[ex['label']]
    return ex

def _rand_sample(lst, frac):
    random.Random(4).shuffle(lst)
    return lst[:int(len(lst)*frac)]