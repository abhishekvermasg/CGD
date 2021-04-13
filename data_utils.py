import argparse
import os

import torch
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd

def process_sop_data():
    train_images, test_images = {}, {}
    data_tuple = {'train': train_images, 'test': test_images}
    
    df_train = pd.read_csv('../input/shopee-clean/train_90.csv')
    df_train = df_train.drop_duplicates(subset=['image'])
    df_train['file_path'] = df_train.image.apply(lambda x: os.path.join(data_dir, x))
    le = LabelEncoder()
    df_train.label_group = le.fit_transform(df_train.label_group)
    gkf = GroupKFold(n_splits=5)
    fold = 0
    df_train['fold'] = -1
    
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(df_train, None, df_train.label_group)):
        df_train.loc[valid_idx, 'fold'] = fold

    for data_type, image_list in data_tuple.items():
        for i in df_train.iterrows():
            label, img_name = i[1]['label_group'], i[1]['image']
            if label in image_list:
                image_list[label].append(save_name)
            else:
                image_list[label] = [save_name]
            if i[1]['fold'] != 0:
                if label in train_images:
                    train_images[label].append(save_name)
                else:
                    train_images[label] = [save_name]
            else:
                if label in test_images:
                    test_images[label].append(save_name)
                else:
                    test_images[label].append(save_name)

    torch.save({'train': train_images, 'test': test_images}, './data_dict.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets')
    parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')

    opt = parser.parse_args()

    print('processing Shopee dataset')
    process_shopee_data()
