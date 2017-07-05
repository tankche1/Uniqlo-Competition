import pandas as pd
import os
from tqdm import *

PATH_TO_TRAIN_IMAGES = os.path.join('data','val')
PATH_TO_TRAIN_DATA = os.path.join('train_master.tsv')
PATH_TO_SAVE=os.path.join('data','val')

def make_observe_data(path_to_train_images, path_to_train_data):
    print('loading train data ...')
    data = pd.read_csv(path_to_train_data, sep='\t')
    X = []
    y = []
    for row in tqdm(data.iterrows()):
        f, l = row[1]['file_name'], row[1]['category_id']
        try:
            if not os.path.exists(os.path.join(PATH_TO_SAVE,str(l))):
                os.mkdir(os.path.join(PATH_TO_SAVE,str(l)))
            
			#print(str(os.path.join(path_to_train_images, f))+str(os.path.join(PATH_TO_SAVE,str(l))))
            os.system('mv '+str(os.path.join(path_to_train_images, f))+' '+str(os.path.join(PATH_TO_SAVE,str(l))))

        except Exception as e:
            print(str(e))

    
    print('done.')

make_observe_data(PATH_TO_TRAIN_IMAGES,PATH_TO_TRAIN_DATA)


