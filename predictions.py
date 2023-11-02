#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re
import string
import warnings
from pretty_html_table import build_table

warnings.filterwarnings('ignore')

### Load NER model
model_ner = spacy.load(r'C:\Users\201311\Desktop\haciz_Dataset\14.10.2023_icra\new_model\model-last')
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

config_tesseract = "--psm 6"


def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('', '', whitespace)
    tablePunctuation = str.maketrans('', '', punctuation)
    text = str(txt)
    # text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)

    return str(removepunctuation)


# group the label
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''

    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id


def parser(text, label):
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D', '', text)

    elif label == 'EMAIL':
        text = text.lower()
        allow_special_char = '@_.\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text)

    elif label == 'WEB':
        text = text.lower()
        allow_special_char = ':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text)

    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        text = text.title()

    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = text.title()

    return text


grp_gen = groupgen()


def tesseract_ocr(img):
    text = pytesseract.image_to_string(img, lang='tur', config=config_tesseract)
    text_1 = text.replace('\n', ' ')
    expr = re.compile('\d{2}/\d{2}/\d{4}')
    line = re.sub(expr, '', text_1)
    line_1 = re.sub(' +', ' ', line)
    print(line_1)
    return line_1


def getPredictions(image):
    df = pd.DataFrame()
    text = tesseract_ocr(image)
    doc1 = model_ner(text)
    list_labels = []
    list_text = []
    list_file_name = []
    for entity in doc1.ents:
        list_labels.append(entity.label_)
        list_text.append(entity.text)
        # list_file_name.append(file_name)
        # print(entity.label_, " — — — ", entity.text)
    df_labels = pd.DataFrame(list_labels, columns=['Labels'])
    df_text = pd.DataFrame(list_text, columns=['Text'])
    # df_doc_name = pd.DataFrame(list_file_name, columns=["Doc Name"])
    frames = [df_labels, df_text]
    df_end = pd.concat(frames, axis=1, join='inner')
    df = df.append(df_end)
    print(text)
    html_table_blue_light = build_table(df, 'blue_light')

    return html_table_blue_light
