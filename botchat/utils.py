# flake8: noqa: F401, F403
import abc
import argparse
import collections
import json
import multiprocessing as mp
import numpy as np
import os, sys, time, base64, io
import os.path as osp
import copy as cp
import pickle
import random as rd
import requests
import shutil
import string
import subprocess
import warnings
import pandas as pd
from collections import OrderedDict, defaultdict
from multiprocessing import Pool, current_process

def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False

def double_log(msg, fout=None):
    print(msg)
    if fout is not None:
        fout.write(str(msg) + '\n')
        fout.flush()

def timestr(second=True, minute=True):
    s = datetime.now().strftime('%Y%m%d%H%M%S')[2:]
    if second:
        return s
    elif minute:
        return s[:-2]
    else:
        return s[:-4]
    
def num2uuid(num):
    rd.seed(num)
    return str(uuid.UUID(int=rd.getrandbits(128), version=4))

def randomuuid():
    seed = rd.randint(0, 2 ** 32 - 1)
    return num2uuid(seed)

def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f

def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))

def default_set(self, args, name, default):
    if hasattr(args, name):
        val = getattr(args, name)
        setattr(self, name, val)
    else:
        setattr(self, name, default)

def dict_merge(dct, merge_dct):
    for k, _ in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def youtube_dl(idx):
    cmd = f'youtube-dl -f best -f mp4 "{idx}"  -o {idx}.mp4'
    os.system(cmd)

def run_command(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd)

def ls(dirname='.', full=True, match=''):
    if not full or dirname == '.':
        ans = os.listdir(dirname)
    ans = [osp.join(dirname, x) for x in os.listdir(dirname)]
    ans = [x for x in ans if match in x]
    return ans

def download_file(url, filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    response = requests.get(url)
    open(filename, 'wb').write(response.content)

# LOAD & DUMP
def dump(data, f):
    def dump_pkl(data, pth):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth):
        json.dump(data, open(pth, 'w'), indent=4)

    def dump_jsonl(data, f):
        lines = [json.dumps(x, ensure_ascii=False) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f):
        data.to_excel(f, index=False)

    def dump_csv(data, f):
        data.to_csv(f, index=False)

    def dump_tsv(data, f):
        data.to_csv(f, sep='\t', index=False)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f)

def load(f):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f) 

