import csv
from os import listdir
from os.path import join

data = dict()
data_dir = 'data/'
dataset_split = 'train'
task = 'task1-span-identification'

for file_name in listdir(join(data_dir, '%s-labels-%s' % (dataset_split, task))):
    file_number = int(''.join(list(filter(str.isdigit, file_name)))) // 10
    with open(r"C:\Users\Burak\PycharmProjects\PropagandaDetection\data\train-labels-task1-span-identification\article" + str(file_number)+ ".task1-SI.labels", 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, row in enumerate(reader):
            if row[2] == 'non-propaganda':
                data[(file_number, int(row[1]))] = [0]
            elif row[2] == 'propaganda':
                data[(file_number, int(row[1]))] = [1]
            else:
                data[(file_number, int(row[1]))] = [-1]

for file_name in listdir(join(data_dir, '%s-articles' % dataset_split)):
    file_number = int(''.join(list(filter(str.isdigit, file_name))))

    with open(join(data_dir, '%s-articles' % dataset_split, file_name), 'r') as file:
        for i, line in enumerate(file):
            data[(file_number, i + 1)].insert(0, line.strip())

with open(join(data_dir, '%s-data.tsv' % dataset_split), 'w') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['file_number', 'line_number', 'text', 'label'])

    for file_number, line_number in data:
        writer.writerow([file_number, line_number] + data[(file_number, line_number)])

