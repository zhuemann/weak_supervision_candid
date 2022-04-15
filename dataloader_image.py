import torch
from torch.utils.data import Dataset
import pydicom as pdcm
import os
from PIL import Image
import numpy as np
from utility import rle_decode_modified

class ImageDataset(Dataset):
    def __init__(self, dataframe, dir_base='D:/candid_ptx/', mode=None, transforms = None, resize = None, img_size = 256): # data_path = os.path.join(dir_base,'Lymphoma_UW_Retrospective/Data/mips/')
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.row_ids = self.data.index
        self.img_size = img_size

        self.df_data = dataframe.values
        self.transforms_augs = transforms
        self.mode = mode
        self.data_path = os.path.join(dir_base, "public_datasets/candid_ptx/dataset1/dataset/")

        self.resize = resize

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        # images data extraction
        img_name = self.row_ids[index]
        img_name = str(img_name) #+ "_mip.png"
        #if exists(os.path.join(self.data_path, 'Group_1_2_3_curated', img_name)):
        #    data_dir = "Group_1_2_3_curated"
        #if exists(os.path.join(self.data_path, 'Group_4_5_curated', img_name)):
        #    data_dir = "Group_4_5_curated"
        data_dir = "public_datasets/candid_ptx/dataset1/dataset/"
        img_path = os.path.join(self.data_path, img_name)
        #print(img_path)

        #print(img_path)
        #DCM_Img = pdcm.read_file(img_path)
        #test = plt.imshow(DCM_Img.pixel_array, cmap=plt.cm.bone)
        #plt.show()


        try:
            DCM_Img = pdcm.read_file(img_path)
            img_raw = DCM_Img.pixel_array
            img_norm = img_raw * (255 / np.amax(img_raw)) # puts the highest value at 255
            img = np.uint8(img_norm)


        except:
            print("can't open")
            print(img_path)

        # decodes the rle
        if self.targets[index] != str(-1):
            segmentation_mask_org = rle_decode_modified(self.targets[index], (1024, 1024))
            segmentation_mask_org = np.uint8(segmentation_mask_org)
        else:
            segmentation_mask_org = np.zeros((1024, 1024))
            segmentation_mask_org = np.uint8(segmentation_mask_org)
        #segmentation_mask_org = Image.fromarray(segmentation_mask_org).convert("RGB")  # makes the segmentation mask into a PIL image
        #segmentation_mask = self.resize(segmentation_mask_org)
        #print(segmentation_mask.size())


        if self.transforms_augs is not None:
            #image = self.transforms(img)
            try:
                if self.mode == "train":
                    #print(type(img))
                    #print(img.shape)
                    img = Image.fromarray(img).convert("RGB")
                    #print(type(img))
                    img = np.array(img)
                    #segmentation_mask_org = np.uint8(segmentation_mask_org)
                    #print(type(img))
                    transformed = self.transforms_augs(image=img, mask=segmentation_mask_org)
                    image = transformed['image']
                    segmentation_mask_org = transformed['mask']
                    image = Image.fromarray(np.uint8(image))  # makes the image into a PIL image
                    image = self.resize(image)  # resizes the image to be the same as the model size
                    #segmentation_mask = Image.fromarray(np.uint8(segmentation_mask))
                    #segmentation_mask = self.resize(segmentation_mask)


                else:
                    img = Image.fromarray(img).convert("RGB")
                    #img = np.array(img)
                    #image = Image.fromarray(img)  # makes the image into a PIL image
                    image = self.transforms_augs(img)
            except:
                print("can't transform")
                print(img_path)
        else:
            image = img

        #print(img.shape)
        #print(segmentation_mask.shape)

        segmentation_mask = Image.fromarray(np.uint8(segmentation_mask_org))
        segmentation_mask = self.resize(segmentation_mask)

        # for showing the images with maps and such
        #plt.figure()
        #DCM_Img = pdcm.read_file(img_path)
        #img_raw = DCM_Img.pixel_array
        #f, ax = plt.subplots(1, 3)
        #ax[0].imshow(img_raw, cmap=plt.cm.bone,)
        #ax[1].imshow(image.squeeze().cpu().detach().numpy(), cmap=plt.cm.bone)
        #ax[2].imshow(segmentation_mask, cmap="jet", alpha = 1)
        #ax[2].imshow(image.squeeze().cpu().detach().numpy(), cmap=plt.cm.bone, alpha = .5)
        #plt.show()
        #print("returing from dataloader")
        return {

            #'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'targets': segmentation_mask,
            'row_ids': self.row_ids[index],
            'images': image
        }