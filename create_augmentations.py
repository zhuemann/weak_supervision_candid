import albumentations as albu
import torchvision.transforms as transforms


def create_augmentations(IMG_SIZE = 256):

    """
    # old augmentatins kept for comparison
    transforms_train = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            #transforms.RandomHorizontalFlip(p=0.3),
            #transforms.RandomVerticalFlip(p=0.3),
            #transforms.RandomAffine(degrees = 10, translate =(.1,.1), scale = None, shear = None),
            #transforms.RandomResizedCrop(IMG_SIZE),
            transforms.PILToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #transforms.Grayscale(num_output_channels=1),
            #transforms.Normalize([0.5], [0.5])
        ]
    )
    """

    # augmentations used in the training step
    albu_augs = albu.Compose([
        #ToTensorV2(),
        albu.HorizontalFlip(),
        albu.OneOf([
            albu.RandomContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
        ], p=.3),  #p=0.3),
        albu.OneOf([
            albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albu.GridDistortion(),
            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=.3),#p=0.3),
        albu.ShiftScaleRotate(),
        #albu.Resize(img_size, img_size, always_apply=True),
    ])

    # augmentations used in the validation and test step
    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.PILToTensor(),
            #transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
            #transforms.Normalize((0.5,), (0.5,))
            #transforms.Grayscale(num_output_channels=1),
            #transforms.Normalize([0.5], [0.5])
        ]
    )

    #Used to resize after some processing in the data loader
    transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])

    return albu_augs, transforms_resize, transforms_valid