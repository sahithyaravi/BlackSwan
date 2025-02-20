import csv
import pandas as pd
import json

with open('../annotations/VAR_Data.json') as f:
    data = json.load(f)

good_list = []
bad_list = []

def find_annot(set_id, annot_id):
    for annot in data:
        if annot['set_id'] == set_id and annot['id'] == annot_id:
            return annot
    return None

def extract_good(listpath):
    with open(listpath) as f:
        df=[tuple(line) for line in csv.reader(f)]
    for row in df:
        set_id = row[0].strip()
        annot_id = row[1].strip()
        annot = find_annot(set_id, annot_id)
        if annot is not None:
            good_list.append(annot['participant_id'])

def extract_bad(listpath):
    with open(listpath) as f:
        df=[tuple(line) for line in csv.reader(f)]
    for row in df:
        set_id = row[0].strip()
        annot_id = row[1].strip()
        annot = find_annot(set_id, annot_id)
        if annot is not None:
            bad_list.append(annot['participant_id'])

extract_good('good-v4.csv')
extract_good('good-v5.csv')
extract_good('good-v6.csv')
extract_good('good-v7.csv')
extract_good('good-v8.csv')

extract_bad('bad-v4.csv')
extract_bad('bad-v5.csv')
extract_bad('bad-v6.csv')
extract_bad('bad-v7.csv')
extract_bad('bad-v8.csv')

good_list = set(good_list)
bad_list = set(bad_list)
good_list -= bad_list
good_list = list(good_list)
bad_list = list(bad_list)

print(','.join(good_list))
print('---- Num: ', len(good_list), ' ----', len(bad_list))
print(','.join(bad_list))

