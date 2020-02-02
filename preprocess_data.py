import csv
from os import listdir
from os.path import join
import re

data = []
data_dir = 'data/'
dataset_split = 'train'
task = 'task1-span-identification'
train_data = True

def clean_text(text):
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


if train_data:
    for file_name in listdir(join(data_dir, '%s-labels-%s' % (dataset_split, task))):
        file_number = int(''.join(list(filter(str.isdigit, file_name)))) // 10
        with open(r"C:\Users\Burak\PycharmProjects\PropagandaDetection\data\train-labels-task1-span-identification\article" + str(file_number)+ ".task1-SI.labels", 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            for i, row in reversed(list(enumerate(reader))):
                with open(r"C:\Users\Burak\PycharmProjects\PropagandaDetection\data\train-articles\article" + str(file_number)+ ".txt", 'r', encoding='utf-8') as file:
                    text = file.read()
                    proptext = clean_text(text[int(row[1]):int(row[2])+1])
                    text = text[:int(row[1])] + text[int(row[2])+1:]
                    data.append([proptext, 1])
                    for i, line in enumerate(text.splitlines()):
                        if len(line) > 20 and not line.isspace():
                            line = clean_text(line)
                            data.append([line, 0])


    with open(join(data_dir, 'data.tsv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        for i, current in data:
            writer.writerow([i, current])


