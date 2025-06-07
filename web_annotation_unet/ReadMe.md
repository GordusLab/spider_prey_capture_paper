# Pipeline for auto web annotation

  **Modifying Abel's U-net from archive_pre25mar2022/old/exploratory/cnn_1/model.py**


 1. Run `predict_web.py` to predict new web from the recordings.
    + Before running the script, remember to convert .avi recordings to .npy
    + It uses  `get_unet.py` to run the Unet
    + The model working well now is `get_unet_acc_n300_modified.keras`
 2. Run `post_annotate.py` if you want to refine the predicted web
  + If the annotation doesn't look good, open flika-spider to annotate those parts and save the ROI as `**_get_roipreymask.npy.txt`
  + Run `post_annotate.py`: this will read the manual annotate web and refine the prediction.

# Unet model
1.  `get_unet.py` is the main code for the Unet structure.
 + This model works well without overfitting issue.
 + I slightly modified the metric = accuracy.


# Getting training dataset
1. `extract_dataset.py` is to extract the manual labeled ROI and the first frame of the recordings. There are totally 91 labeled data.

# Train the network
1. `main.py` is the main script to train the network

 + In this script, it randomly erase 10 (=size) different region of the web.
 + Repeated the erasing process 30 times (N) for each image. Therefore, we got 30 different image for each web.
 + Do the erasing for all 91 images. Therefore, we expand the training dataset from 91 to 91*31=2821 images.
