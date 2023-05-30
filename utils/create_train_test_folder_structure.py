import shutil
import os
import pandas as pd
import numpy as np

def create_train_test_folder_structure(n_classes, path):
    
    # read in the list of celebreties left in the folder
    if not os.path.isfile("{}/Data/identity_CelebA_updated.txt".format(path)):
        identity_keys = pd.read_csv("{}/Data/identity_CelebA.txt".format(path), sep=" ", header=None, names = ["file_name", "pict_id"])
    else:
        identity_keys = pd.read_csv("{}/Data/identity_CelebA_updated.txt".format(path), sep = " ") 
    identity_key_updated = identity_keys
    
    # create train & test folders by moving images in to their respective classes
    for n_id in range(n_classes):
        x = identity_keys.pict_id[n_id]
        identity_x = identity_keys[identity_keys.pict_id == x].reset_index(drop = True)

        
        for n_img in range(len(identity_x)):
            id_x_file = identity_x.iloc[n_img,0][0:-4]

            path_original = "{}\Data\img_align_celeba\img_align_celeba_png\{}.png".format(path,id_x_file)
            path_target = "{}/Data/img_align_celeba/class_data/{}/".format(path, x)

            # make sure target directory exists
            if not os.path.exists(path_target):
                os.makedirs(path_target)
            try:
                shutil.move(path_original, path_target)
            except:
                FileNotFoundError
                print("File Not Found: {}".format(id_x_file))
                pass
        identity_key_updated = identity_key_updated[identity_key_updated.pict_id != x]

    # save an updated list of celebreties left in the original folder
    identity_key_updated.to_csv("./Data/identity_CelebA_updated.txt", sep = " ", index = False)