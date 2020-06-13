import csv
import os
import random
import shutil
from tqdm import tqdm

# Read all files from train folder
# randomly choose 1/5 of it for validation, and rest for training
# copy images into new location

TRAIN_CSV_PATH = '/Users/elimgta/dev/global-wheat-detection/train.csv'
INPUT_IMAGE_PATH = '/Users/elimgta/dev/global-wheat-detection/train/'
WORKING_PATH = '/Users/elimgta/dev/global-wheat-detection/'
RANDOM_SEED = 11

all_images = os.listdir(INPUT_IMAGE_PATH)
print(len(all_images))
all_images = list(filter(lambda x: not os.path.isfile(x), all_images))
random.seed(RANDOM_SEED)

random.shuffle(all_images)
print(len(all_images))
print(all_images[:10])
sep = int(len(all_images) / 5)
val_images = all_images[:sep]
train_images = all_images[sep:]
print(len(val_images) + len(train_images))

try:
    shutil.rmtree(os.path.join(WORKING_PATH, 'split'))
except:
    pass
os.makedirs(os.path.join(WORKING_PATH, 'split/val'), exist_ok=True)
os.makedirs(os.path.join(WORKING_PATH, 'split/train'), exist_ok=True)
for fname in val_images:
    shutil.copyfile(os.path.join(INPUT_IMAGE_PATH, fname), os.path.join(WORKING_PATH, 'split/val/', fname))
for fname in train_images:
    shutil.copyfile(os.path.join(INPUT_IMAGE_PATH, fname), os.path.join(WORKING_PATH, 'split/train/', fname))

labels = {}
with open(TRAIN_CSV_PATH) as csvfile:
    reader = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
    next(reader)
    for row in tqdm(reader):
        if row[0] not in labels:
            labels[row[0]] = []
        width = int(row[1])
        height = int(row[2])
        box = eval(row[3])
        # box -- (xmin, ymin, width, height)
        labels[row[0]].append((0, (box[0] + box[2] / 2.) / width, (box[1] + box[3] / 2.) / height, box[2] / width, box[3] / height))

print(len(labels))
try:
    shutil.rmtree(os.path.join(WORKING_PATH, 'labels'))
except:
    pass
os.makedirs(os.path.join(WORKING_PATH, 'labels/val'), exist_ok=True)
os.makedirs(os.path.join(WORKING_PATH, 'labels/train'), exist_ok=True)
val_images = set([os.path.splitext(fname)[0] for fname in val_images])
train_images = set([os.path.splitext(fname)[0] for fname in train_images])
print(list(val_images)[:10])
print(list(train_images)[:10])
# Images without any rectangles need to be explicitly added.
for fname in val_images:
    if fname not in labels:
        labels[fname] = []
for fname in train_images:
    if fname not in labels:
        labels[fname] = []
for k, lst in labels.items():
    dst_type = None
    if k in val_images:
        dst_type = 'val'
    elif k in train_images:
        dst_type = 'train'
    else:
        print('CSV row without corresponding image: %s' % k)
        continue
    with open(os.path.join(WORKING_PATH, 'labels/%s/%s.txt' % (dst_type, k)), 'w') as f:
        for v in lst:
            f.write('%d %0.6f %0.6f %0.6f %0.6f\n' % (v[0], v[1], v[2], v[3], v[4]))
