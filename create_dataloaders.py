import os
import pandas as pd
from sklearn import model_selection
from torch.utils.data import DataLoader


from create_augmentations import create_augmentations
from dataloader_image import ImageDataset



def create_dataloaders(config):
    dir_base = config["dir_base"]
    seed = config["seed"]
    IMG_SIZE = config["IMG_SIZE"]
    batch_size = config["batch_size"]
    data_path = config["data_path"] = "D:/candid_ptx/"
    train_samples = config["train_samples"]
    test_samples = config["test_samples"]

    print(dir_base)

    dataframe_location = os.path.join(dir_base,
                                      'Zach_Analysis/candid_data/pneumothorax_large_df.xlsx')  # pneumothorax_df chest_tube_df rib_fracture

    dataframe_location = os.path.join(dir_base,
                                      'Zach_Analysis/candid_data/pneumothorax_large_df.xlsx')
    dataframe_location = os.path.join(data_path, 'pneumothorax_large_df.xlsx' )

    # gets the candid labels and saves it off to the location
    # df = get_candid_labels(dir_base=dir_base)
    # df.to_excel(dataframe_location, index=False)

    # reads in the dataframe as it doesn't really change to save time
    df = pd.read_excel(dataframe_location, engine='openpyxl')
    print(df)

    df.set_index("image_id", inplace=True)
    print(df)


    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, train_size=train_samples, random_state=seed, shuffle=True  # stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=test_samples, random_state=seed, shuffle=True  # stratify=test_valid_df.label.values
    )

    #test_dataframe_location = os.path.join(dir_base, 'Zach_Analysis/candid_data/pneumothorax_df_testset.xlsx')
    #test_df.to_excel(test_dataframe_location, index=True)

    print("train_df")
    print(train_df)

    albu_augs, transforms_resize, transforms_valid = create_augmentations()



    training_set = ImageDataset(dataframe= train_df, mode="train", transforms=albu_augs,
                                    resize=transforms_resize, dir_base=dir_base, img_size=IMG_SIZE)
    valid_set = ImageDataset(dataframe= valid_df, transforms=transforms_valid, resize=transforms_resize,
                                 dir_base=dir_base, img_size=IMG_SIZE)
    test_set = ImageDataset(dataframe= test_df, transforms=transforms_valid, resize=transforms_resize,
                                dir_base=dir_base, img_size=IMG_SIZE)

    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 4
                    }

    test_params = {'batch_size': batch_size,
                   'shuffle': True,
                   'num_workers': 4
                   }

    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **test_params)
    test_loader = DataLoader(test_set, **test_params)


    return training_loader, valid_loader, test_loader