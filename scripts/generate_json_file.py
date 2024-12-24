import os
import json

# datasetdir = r"E:\ABIDE_Dataset\ABIDE_CPAC\func_preproc"
# phenotypicfile = r"E:\ABIDE_Dataset\Phenotypic_V1_0b.csv"

datasetdir = "/data/xuruipeng/ABIDE_CPAC/func_preproc"
phenotypicfile = "/data/xuruipeng/ABIDE_CPAC/Phenotypic_V1_0b.csv"

def get_phenotypic_data():
    with open(phenotypicfile, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    data = []
    subIndex = {}
    for i, line in enumerate(lines[1:]):
        line = line.strip().split(',')
        data.append(dict(zip(header, line)))
        subIndex[line[1]] = i
    return data, header, lines[1:], subIndex
# print(len(get_phenotypic_data()[0]))
# print(get_phenotypic_data()[0][:1])
# print(get_phenotypic_data()[1])
# print(get_phenotypic_data()[2][0])
# print(get_phenotypic_data()[3])
data, header, lines, subIndex = get_phenotypic_data()

def get_dataset():
    dataset = []

    for root, dirs, files in os.walk(datasetdir):
        for file in files:
            if file.endswith('.nii.gz'):
                dataset.append(os.path.join(root, file))
    return dataset

print(get_dataset()[:5])

dataset = []
for i, item in enumerate(get_dataset()):
    print(i, item)

    index = item.split('_')[-3][2:]
    print(index)
    cat = data[subIndex[index]]
    cat['filepath'] = item
    dataset.append(cat)
    # break

# print(dataset[:5])
# print(dataset[0])
# if dataset[0]['SRS_COMMUNICATION'] is not '':
#     print('Not None', dataset[0]['SRS_COMMUNICATION'])
# else:
#     print('None')

# write
with open('../datas/ABIDE_CPAC_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4)

print('\n\n\nTotal number of samples:', len(dataset))