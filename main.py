# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from segmentatin_training import segmentation_training

# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    # Sets which directory to use point this to the folder with the candid data
    local = True
    if local == True:
        directory_base = "Z:/"
        # directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"
    else:
        directory_base = "/UserData/"


    config = {}
    config["seed"] = 1
    config["batch_size"] = 8
    config["dir_base"] = directory_base
    config["epochs"] = 150
    config["n_classes"] = 2
    config["LR"] = 1e-5
    config["IMG_SIZE"] = 256
    config["train_samples"] = 120
    config["test_samples"] = 120
    # should point to you external hard drive with data or wherever you move it
    config["data_path"] = "D:/candid_ptx/"

    acc, valid_log = segmentation_training(config)