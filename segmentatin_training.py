import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import numpy as np
import gc
from tqdm import tqdm

from create_dataloaders import create_dataloaders
from utility import dice_coeff


def segmentation_training(config = None):

    IMG_SIZE = config["IMG_SIZE"]
    BATCH_SIZE = config["batch_size"]
    LR = 1e-5  # 8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    N_EPOCHS = config["epochs"]
    N_CLASS = config["n_classes"]

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

    output_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))])

    best_acc = -1
    valid_log = []
    for epoch in range(1, N_EPOCHS + 1):
        model_obj.train()
        gc.collect()
        training_dice = []

        for _, data in tqdm(enumerate(training_loader, 0)):
            targets = data['targets'].to(device, dtype=torch.float)
            targets = torch.squeeze(targets)
            images = data['images'].to(device, dtype=torch.float)

            # print(images.shape)
            # outputs = model_obj(ids, mask, token_type_ids, images)
            outputs = model_obj(images)
            # print(type(outputs))
            outputs = output_resize(torch.squeeze(outputs, dim=1))

            optimizer.zero_grad()
            # loss = loss_fn(outputs[:, 0], targets)
            loss = criterion(outputs, targets)
            # print(loss)
            if _ % 20 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

                # f, ax = plt.subplots(1, 3)
                # ax[0].imshow(np.uint8(torch.permute(images[0], (1,2,0)).squeeze().cpu().detach().numpy()))
                # ax[1].imshow(outputs[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                # ax[2].imshow(targets[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                # ax[2].imshow(np.uint8(torch.permute(images[0], (1,2,0)).squeeze().cpu().detach().numpy()), cmap=plt.cm.bone, alpha=.5)

                # out_img = plt.imshow(outputs[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                # plt.show()
                # tar_img = plt.imshow(targets[0].squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
                # plt.show()

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
