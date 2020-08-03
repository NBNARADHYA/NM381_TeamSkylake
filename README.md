# SIH2020_NM381_TeamSkylake
Official code repository of Team Skylake for SIH 2020 grand finale.

## FINAL PRESENTATION
https://docs.google.com/presentation/d/1o55oXnzfp19Sy6HjDW_XEgdmVyPkc0L--XAyHjT4BzI/edit#slide=id.p3!

#### [Android app apk](https://drive.google.com/file/d/17NIiGfxXryZxkhsEZoDY4JC0C-JDRMxz/view?usp=sharing)


### DEMO
![](demo.gif)


### DATA SET USED
A large dataset of webcam images annotated with sky regions(90,000)
  **SOURCE:** Nathan Jacobs Group

### DATA PREPROCESSING
Dataset consists of many corrupted images, so we wrote our own python scripts to remove those corrupted images.
### OTHER PREPROCESSING TECHNIQUES
     → Random Rotation
     → Gaussian Blur
     → Normalization
            
### MODELS USED
UNET model with RESNET34 encoder
### LOSS FUNCTION USED
Weighted average of Soft Dice
Focal Loss
### METRIC OF EVALUATION
IOU (Intersection over Union)

### TRAINING APPROACH
     → used model pre trained on IMAGENET.
     → progressive training due to huge size of data set
     → started with 20 percent  of dataset to provide warm start for training
     → Gradually increased to 70 percent  in step of 5
     → this was used along with 5 fold cross validation due to the lack of diversity in images
     → tested on 30 percent images
### IOU on Validation   : → 0.9959
### IOU on Test         : → 0.9835

