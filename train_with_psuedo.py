import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import numpy as np
import gc
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle

from create_dataloaders import create_dataloaders
from labeling_func_prediction import average_labeling_prediction, labeling_func_prediction, get_psuedo_label
from utility import dice_coeff

def train_with_psuedo_labels(config = None, weight1=None, weight2=None, weight3=None):


    IMG_SIZE = config["IMG_SIZE"]
    BATCH_SIZE = 4
    LR = 1e-5  # 8e-5
    N_EPOCHS = config["epochs"]
    N_CLASS = config["n_classes"]
    dir_base = config["dir_base"]

    # gets the dataloaders, you should have df_location be the path to the df which you want predictions for
    # you should change df_location to be the dataset with roughly 2500 images and should look something like
    weak_datapath = os.path.join(dir_base,'Zach_Analysis/candid_data/weak_supervision/weak_supervision_trainset_df.xlsx')
    training_loader, valid_loader, test_loader = create_dataloaders(config, df_location=weak_datapath)

    # gets the model which are random and loads in the trained weights to give us five labeling fucntions
    model1 = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    model2 = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    model3 = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    #model4 = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    final_model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)


    # sets up the paths for each model to be loaded from this path should point to the folder with the models in it
    model_base = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/weak_supervision_models/roberta_labeling_functions/')
    model1_path = model_base + "segmentation_candid42"
    model2_path = model_base + "segmentation_candid117"
    model3_path = model_base + "segmentation_candid295"
    #model4_path = model_base + "segmentation_candid456"
    #model5_path = model_base + "segmentation_candid712"

    # loads in the weights for each model give the path specified above
    model1.load_state_dict(torch.load(model1_path, map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load(model2_path, map_location=torch.device('cpu')))
    model3.load_state_dict(torch.load(model3_path, map_location=torch.device('cpu')))
    #model4.load_state_dict(torch.load(model4_path, map_location=torch.device('cpu')))
    #model5.load_state_dict(torch.load(model5_path, map_location=torch.device('cpu')))


    # sends all the models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model2.to(device)
    model3.to(device)

    final_model.to(device)


    # for resizing the output from the model back down to our label size
    output_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))])


    # loss function for this segmentation task
    #criterion = nn.BCEWithLogitsLoss()

    #optimizer = torch.optim.Adam(params=final_model.parameters(), lr=LR)

    dice_weight = []
    dice_avg = []

    for _, data in tqdm(enumerate(training_loader, 0)):
        # gets the images and labels from the data loader
        targets = data['targets'].to(device, dtype=torch.float)
        targets = torch.squeeze(targets)
        images = data['images'].to(device, dtype=torch.float)

        # goes through the model and resized
        output1 = model1(images)
        output1 = output_resize(torch.squeeze(output1, dim=1))

        output2 = model2(images)
        output2 = output_resize(torch.squeeze(output2, dim=1))
        output3 = model3(images)
        output3 = output_resize(torch.squeeze(output3, dim=1))

        #these just put each output through a sigmoid and round it to gets its prediction of 1 or 0 for each pixel
        sigmoid = torch.sigmoid(output1)
        output1 = torch.round(sigmoid)

        sigmoid = torch.sigmoid(output2)
        output2 = torch.round(sigmoid)
        sigmoid = torch.sigmoid(output3)
        output3 = torch.round(sigmoid)


        #weak_label = get_psuedo_label(weight1, weight2, weight3, output1, output1, output1)

        #d1 = dice_coeff(torch.squeeze(weak_label), targets)
        #d1 = d1.cpu().detach().numpy()
        #dice_weight.append(d1)

        avg_output = average_labeling_prediction(output1, output2, output3)
        print(avg_output.shape)
        d_avg = dice_coeff(avg_output, targets)
        print(d_avg)
        d_avg.cpu().detach().numpy()
        print(d_avg)
        dice_avg.append(d_avg)


    #print(f"Weighting Scheme:  {np.mean(dice_weight)}")
    print(f"Averaging: {np.mean(dice_avg)}")
