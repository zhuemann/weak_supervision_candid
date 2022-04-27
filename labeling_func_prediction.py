import torch
from create_dataloaders import create_dataloaders

def labeling_func_prediction(output1, output2, output3, output4, output5):
    """
    This is what needs to be implemented from the paper

    :param output1-5: The pixel wise prediction from each model for a given image

    :return: The combined prediction given the five model outputs
    """


    return None



def average_labeling_prediction(output1, output2, output3, output4, output5):

    pooled_outputs = output1 + output2 +output3 + output4 + output5

    pooled_outputs = torch.round(pooled_outputs/5)

    return pooled_outputs
