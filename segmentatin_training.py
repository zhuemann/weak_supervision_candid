import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import numpy as np
import gc
from tqdm import tqdm
import os

from create_dataloaders import create_dataloaders
from utility import dice_coeff
from plot_images import plot_images


def segmentation_training(config = None):

    IMG_SIZE = config["IMG_SIZE"]
    BATCH_SIZE = config["batch_size"]
    LR = 1e-5  # 8e-5
    N_EPOCHS = config["epochs"]
    N_CLASS = config["n_classes"]
    dir_base = config["dir_base"]

    # gets the dataloaders, hopefully you won't have to play with these functions but good luck if you do have to
    training_loader, valid_loader, test_loader = create_dataloaders(config)


    # gets the model which is initialized with imagenet weights, you can change to None to have random initialization
    model_obj = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)

    # if you have a gpu use it else use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obj.to(device)

    # loss function for this segmentation task
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=LR)

    # for resizing the output from the model back down to our label size
    output_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))])

    # set the save path to where ever you want the model to be saved to
    save_path = os.path.join(dir_base,'Zach_Analysis/models/candid_finetuned_segmentation/forked_1/segmentation_forked_candid')

    best_acc = -1
    valid_log = []
    for epoch in range(1, N_EPOCHS + 1):
        model_obj.train()
        gc.collect()
        training_dice = []

        for _, data in tqdm(enumerate(training_loader, 0)):
            # gets the images and labels from the data loader
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            # goes through the model and resized
            outputs = model_obj(images)
            outputs = output_resize(torch.squeeze(outputs, dim=1))

            optimizer.zero_grad()
            loss = criterion(outputs, targets)

            if _ % 20 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

                # if you want to plot some images use this function
                # plot_images(images, targets, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)

            # calculates the dice coefficent for each image and adds it to the list
            for i in range(0, outputs.shape[0]):
                dice = dice_coeff(outputs[i], targets[i])
                dice = dice.item()
                training_dice.append(dice)

        avg_training_dice = np.average(training_dice)
        print(f"Epoch {str(epoch)}, Average Training Dice Score = {avg_training_dice}")

        # each epoch, look at validation data

        with torch.no_grad():
            model_obj.eval()
            valid_dice = []
            gc.collect()
            for _, data in tqdm(enumerate(valid_loader, 0)):
                # gets the images and labels from the data loader
                targets = data['targets'].to(device, dtype=torch.float)
                images = data['images'].to(device, dtype=torch.float)

                # goes through the model and resized
                outputs = model_obj(images)
                outputs = output_resize(torch.squeeze(outputs, dim=1))

                # put output between 0 and 1 and rounds to nearest integer ie 0 or 1 labels
                sigmoid = torch.sigmoid(outputs)
                outputs = torch.round(sigmoid)

                # calculates the dice coefficent for each image and adds it to the list
                for i in range(0, outputs.shape[0]):
                    dice = dice_coeff(outputs[i], targets[i])
                    dice = dice.item()
                    valid_dice.append(dice)

            avg_valid_dice = np.average(valid_dice)
            print(f"Epoch {str(epoch)}, Average Valid Dice Score = {avg_valid_dice}")
            valid_log.append(avg_valid_dice)

            # saves the model with the highest validation dice score
            if avg_valid_dice >= best_acc:
                best_acc = avg_valid_dice
                torch.save(model_obj.state_dict(), save_path)

    model_obj.eval()

    # load the best model in from the validation step and evaluate on it
    model_obj.load_state_dict(torch.load(save_path))

    with torch.no_grad():
        test_dice = []
        gc.collect()
        for _, data in tqdm(enumerate(test_loader, 0)):
            # gets the images and labels from the data loader
            targets = data['targets'].to(device, dtype=torch.float)
            images = data['images'].to(device, dtype=torch.float)

            # goes through the model and resized
            outputs = model_obj(images)
            outputs = output_resize(torch.squeeze(outputs, dim=1))

            # threshold each pixel for prediction of 0 or 1
            sigmoid = torch.sigmoid(outputs)
            outputs = torch.round(sigmoid)

            # calculates the dice score for all each image as they are output
            for i in range(0, outputs.shape[0]):
                dice = dice_coeff(outputs[i], targets[i])
                dice = dice.item()
                test_dice.append(dice)

        avg_test_dice = np.average(test_dice)
        print(f"Epoch {str(epoch)}, Average Test Dice Score = {avg_test_dice}")


        return avg_test_dice, valid_log