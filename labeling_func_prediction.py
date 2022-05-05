import torch
from create_dataloaders import create_dataloaders
import pickle
import numpy as np
from numpy import linalg as LA

def labeling_func_prediction(pred1, pred2, pred3):
#def labeling_func_prediction():
    """
    This is what needs to be implemented from the paper

    :param output1-5: The pixel wise prediction from each model for a given image

    :return: The combined prediction given the five model outputs
    """
    """
    print("insdie stuff")
    with open('pred1.pickle', 'rb') as handle:
        pred1 = pickle.load(handle)

    with open('pred2.pickle', 'rb') as handle:
        pred2 = pickle.load(handle)

    with open('pred3.pickle', 'rb') as handle:
        pred3 = pickle.load(handle)
    """


    #dif = np.zeros(256,256)
    emprical_error12 = 0
    emprical_error13 = 0
    emprical_error23 = 0



    for i in range(0, len(pred1)):

        dif12 = pred1[i] - pred2[i]
        emprical_error12 += LA.norm(dif12)

        dif13 = pred1[i] - pred3[i]
        emprical_error13 += LA.norm(dif13)

        dif23 = pred2[i] - pred3[i]
        emprical_error23 += LA.norm(dif23)


    print(emprical_error12/len(pred1))
    print(emprical_error13/len(pred1))
    print(emprical_error23/len(pred1))


    weight1 = (1/2)*( (emprical_error12/len(pred1)) + (emprical_error13/len(pred1)) - (emprical_error23/len(pred1)) )
    weight2 = (1/2)*( (emprical_error12/len(pred1)) + (emprical_error23/len(pred1)) - (emprical_error13/len(pred1)) )
    weight3 = (1/2)*( (emprical_error13/len(pred1)) + (emprical_error23/len(pred1)) - (emprical_error12/len(pred1)) )
    print(len(pred1))


    print(weight1)
    print(weight2)
    print(weight3)


    return weight1, weight2, weight3



def average_labeling_prediction(output1, output2, output3, output4, output5):

    pooled_outputs = output1 + output2 +output3 + output4 + output5

    pooled_outputs = torch.round(pooled_outputs/5)

    return pooled_outputs


def get_psuedo_label(weight1, weight2, weight3, mask1, mask2, mask3):

    pooled_masks = weight1*mask1 + weight2*mask2 + weight3*mask3

    label = torch.rand(pooled_masks)

    print(label)
    return label
