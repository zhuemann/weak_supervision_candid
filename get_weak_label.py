import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import numpy as np
import gc
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from create_dataloaders import create_dataloaders
from labeling_func_prediction import average_labeling_prediction, labeling_func_prediction
from utility import dice_coeff

def get_weak_label(config = None):


    IMG_SIZE = config["IMG_SIZE"]
    BATCH_SIZE = 1
    LR = 1e-5  # 8e-5
    N_EPOCHS = config["epochs"]
    N_CLASS = config["n_classes"]
    dir_base = config["dir_base"]

    # gets the dataloaders, you should have df_location be the path to the df which you want predictions for
    # you should change df_location to be the dataset with roughly 2500 images and should look something like
    # os.path.join(dir_base,'Zach_Analysis/candid_data/pneumothorax_testset_df.xlsx')
    training_loader, valid_loader, test_loader = create_dataloaders(config, df_location=None)

    # gets the model which are random and loads in the trained weights to give us five labeling fucntions
    model1 = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    model2 = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    model3 = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    model4 = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    model5 = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)

    # sets up the paths for each model to be loaded from this path should point to the folder with the models in it
    model_base = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/weak_supervision_models/')
    model1_path = model_base + "segmentation_candid98"
    model2_path = model_base + "segmentation_candid117"
    model3_path = model_base + "segmentation_candid295"
    model4_path = model_base + "segmentation_candid456"
    model5_path = model_base + "segmentation_candid712"

    # loads in the weights for each model give the path specified above
    model1.load_state_dict(torch.load(model1_path, map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load(model2_path, map_location=torch.device('cpu')))
    model3.load_state_dict(torch.load(model3_path, map_location=torch.device('cpu')))
    model4.load_state_dict(torch.load(model4_path, map_location=torch.device('cpu')))
    model5.load_state_dict(torch.load(model5_path, map_location=torch.device('cpu')))

    # sends all the models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model2.to(device)
    model3.to(device)
    model4.to(device)
    model5.to(device)

    # for resizing the output from the model back down to our label size
    output_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))])

    for _, data in tqdm(enumerate(test_loader, 0)):
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
        output4 = model4(images)
        output4 = output_resize(torch.squeeze(output4, dim=1))
        output5 = model5(images)
        output5 = output_resize(torch.squeeze(output5, dim=1))

        #these just put each output through a sigmoid and round it to gets its prediction of 1 or 0 for each pixel
        sigmoid = torch.sigmoid(output1)
        output1 = torch.round(sigmoid)
        sigmoid = torch.sigmoid(output2)
        output2 = torch.round(sigmoid)
        sigmoid = torch.sigmoid(output3)
        output3 = torch.round(sigmoid)
        sigmoid = torch.sigmoid(output4)
        output4 = torch.round(sigmoid)
        sigmoid = torch.sigmoid(output5)
        output5 = torch.round(sigmoid)

        # does an average voting for each pixel
        avg_output = average_labeling_prediction(output1, output2, output3, output4, output5)

        # not implemented but this is where we will do the combination of the models
        ising_output = labeling_func_prediction(output1, output2, output3, output4, output5)

        # calculates the accuracy metric for each model prediction and the average we found
        d1 = dice_coeff(torch.squeeze(output1), targets)
        d2 = dice_coeff(torch.squeeze(output2), targets)
        d3 = dice_coeff(torch.squeeze(output3), targets)
        d4 = dice_coeff(torch.squeeze(output4), targets)
        d5 = dice_coeff(torch.squeeze(output5), targets)
        d_avg = dice_coeff(avg_output, targets)
        f, ax = plt.subplots(1, 8)

        # plots all the outputs with the dice score as the title for each of the models, the average, the true label, and the input
        ax[0].imshow(output1[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
        ax[0].set_title(str(d1.item())[0:6], size=10)
        ax[1].imshow(output2[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
        ax[1].set_title(str(d2.item())[0:6], size=10)
        ax[2].imshow(output3[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
        ax[2].set_title(str(d3.item())[0:6], size=10)
        ax[3].imshow(output4[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
        ax[3].set_title(str(d4.item())[0:6], size=10)
        ax[4].imshow(output5[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
        ax[4].set_title(str(d5.item())[0:6], size=10)
        ax[5].imshow(avg_output[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
        ax[5].set_title("simple_average: " + str(d_avg.item())[0:6], size=10)
        ax[6].imshow(targets.squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
        ax[6].set_title("target", size=10)
        ax[7].imshow(np.uint8(torch.permute(images[0], (1, 2, 0)).squeeze().cpu().detach().numpy()))


        plt.show()