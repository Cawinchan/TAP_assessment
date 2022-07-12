import glob
from PIL import Image
import pandas as pd
import os
import shutil

from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split

def convert_all_images_from_png_to_jpg(IMG_DIR) -> None:
    """
    Converts all .png files to .jpg through PIL .convert("RGB")
    :param IMG_DIR: str

    """
    # keep count on the number of .png files has been coverted 
    counter = 0 
    for path in glob.glob(f"{IMG_DIR}/*.png"):
            try:
                im = Image.open(path)
                rgb_im = im.convert("RGB")
                rgb_im.save(path.replace('.png','.jpg'))
                counter += 1
            except:
                # raises error if a .png image failed to convert, guarentees all .png files convert to .jpg
                raise Exception(f"Unable to convert image {path}")
    if counter > 0:
        print("Successfully converted {counter} .png images")
    else:
        print("No .png images found")

def build_train_df(IMG_DIR,LABEL_FILENAME):
    """
    Creates a pandas DataFrame and finds the intersection between labels available in .csv file and samples given
    :param IMG_DIR: str
    :param LABEL_FILENAME: str
    :returns train_df: pd.DataFrame()

    """
    # reads CSV file
    train_labels_df = pd.read_csv(f"{IMG_DIR}/{LABEL_FILENAME}")
    print(train_labels_df)
    # Note if using Linux distribution use f"{IMG_DIR}/" instead of f"{IMG_DIR}\\"
    # create df for train samples
    train_samples_df = pd.DataFrame([path.replace(f"{IMG_DIR}\\","") for path in glob.glob(f"{IMG_DIR}/*.jpg")],columns=['image'])
    print(train_samples_df)
    # keeps rows where image sample name exists in both train_labels_df and train_samples_df
    train_df = pd.merge(train_labels_df, train_samples_df, how ='inner', on =['image'])
    return train_df

def process_data(IMG_DIR,train_df,dir,classes):
    """
    Checks if the data is in the correct format for torchvision's ImageFolder
    root/dog/xxx.png
    root/dog/xxy.png

    root/cat/123.png
    root/cat/nsdf3.png
    
    """
    # Create a new directory, if exist raise error
    os.makedirs(f"{dir}/processed_data/", exist_ok=False)
    # Finds all the folder names in the processed_data directory
    img_dir_folders = next(os.walk(f"{dir}/processed_data/"))[1]
    # get target classes that we should see in processed_data directory
    classes = [str(i) for i in classes]
    # Create image label key pair {'2810798.jpg': 0, '2818929.jpg': 0}
    train_dic = train_df.set_index('image').to_dict()['category']
    # Checks if there are any existing proccesed data folders and only add missing folders (Reduce replacing of data)
    print(set(img_dir_folders),set(classes))
    if set(img_dir_folders) != set(classes):
        for j in classes:
                if j not in img_dir_folders:
                        # If {dir}/process_data/target_name does not exist, create it
                        os.makedirs(f"{dir}/processed_data/{j}", exist_ok=False)
                # For each image in category, copy it in its respective target_name folder
                for img_n in train_df.loc[train_df.category == int(j)].image:
                        try:
                                # Copy images over
                                shutil.copy(src=f"{IMG_DIR}/{img_n}", dst=f"{IMG_DIR}/processed_data/{train_dic[img_n]}/{img_n}")
                        except KeyError:
                                # Guarentees completeness as all images must be able to be copied
                                raise Exception(f"Unable to copy {img_n}")   


# We want to ensure that our validation dataset has the same distribution as our training dataset
# As such we will build a sampler for pytorch's dataloader
def stratified_random_shuffle_sampler(train_dataset,test_size=0.3,shuffle=True):
    train_idx, valid_idx= train_test_split(
                                            np.arange(len(train_dataset.targets)),
                                            test_size=test_size,
                                            shuffle=shuffle,
                                            stratify=train_dataset.targets
                                            )
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler
    
