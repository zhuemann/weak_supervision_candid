# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from segmentatin_training import segmentation_training
from get_weak_label import get_weak_label
from labeling_func_prediction import labeling_func_prediction
from train_with_psuedo import train_with_psuedo_labels

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # Sets which directory to use point this to the folder with the candid data
    local = False
    if local == True:
        directory_base = "Z:/"
        # directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"
    else:
        directory_base = "/UserData/"


    config = {}
    config["seed"] = 1
    config["batch_size"] = 1#8
    config["dir_base"] = directory_base
    config["epochs"] = 150
    config["n_classes"] = 2
    config["LR"] = 1e-5
    config["IMG_SIZE"] = 512
    config["train_samples"] = 1
    config["test_samples"] = 1
    # should point to you external hard drive with data or wherever you move it
    config["data_path"] = "D:/candid_ptx/"

    pred1, pred2, pred3 = get_weak_label(config=config)
    weight1, weight2, weight3 = labeling_func_prediction(pred1, pred2, pred3)
    weight1 = 1/weight1
    weight2 = 1/weight2
    weight3 = 1/weight3
    norm_factor = 1/(weight1 + weight2+ weight3)
    weight1 = norm_factor*weight1
    weight2 = norm_factor*weight2
    weight3 = norm_factor*weight3

    train_with_psuedo_labels(config, weight1, weight2, weight3)

    #acc, valid_log = segmentation_training(config)
    #get_weak_label(config=config)