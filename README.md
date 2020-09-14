# COVID-19-CT-Image-Diagnosis-via-Transfer-Learning


## Introduction
This project is designed to use transfer learning to classify COVID-19 by lung CT scan. The pre-train  weight and the following models in [Keras applications](https://keras.io/api/applications/) is being applied :

- ResNet50V2 
- Xception 
- DenseNet201 
- MobileNetV2


## Data 
This project is originally designed for "INFORMS 2020 QSR Data Challenge - CT Scan Diagnosis for COVID-19". The dataset is provided by competition organizer. To accesss dataset, you can follow the guildline in [challenge website](https://connect.informs.org/communities/community-home/digestviewer/viewthread?MessageKey=d8770470-40c4-4662-b8ca-d052fa17aaf8&CommunityKey=1d5653fa-85c8-46b3-8176-869b140e5e3c&tab=digestviewer) or [here](https://connect.informs.org/HigherLogic/System/DownloadDocumentFile.ashx?DocumentFileKey=953f3ec3-7d2d-9097-de0c-231d9b820505).

Number of COVID    : 251   
Number of NonCOVID : 292   
Total data         : 543


<p align='center'>
<img src= 'Images/covid_img.png' alt= 'covid_img' height= 224px width= 224px style="padding:20px"/><img src= 'Images/noncovid_img.jpg' alt= 'non_covid_img' height= 224px width= 224px style="padding:20px"/>
    <br>CT COVID(left), CT Non-COVID(right)
</p>




## How to run
#### Customize your input
```python3
def load_train():
    dir = "Images-processed"                        # your file directory
    covid_dir = dir+"/CT_COVID/"
    noncovid_dir = dir+"/CT_NonCOVID/"
```

```python3
def estimate(X_train, y_train, back_bone):          
    IMAGE_WIDTH = 224                               # Image width
    IMAGE_HEIGHT = 224                              # Image height
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)    # (width, height, channel) channel = 3 ---> RGB
    batch_size = 8                                  # Batch size 
    epochs = 40                                     # Number of epochs
    ntrain = 0.8 * len(X_train)                     # Split data with 80/20 train/validation 
    nval = 0.2 * len(X_train)

    X_train, X_val, y_train, y_val = train_test_split(  # 20% validation set
        x, y_train, test_size=0.20, random_state=2)
```
#### Run model
Step 1. Execute Model.py in terminal.

```bash
python3 Model.py
```
Step 2. Select transfer learning model by input model's name. For example, in the following, we choose ResNet50V2 as the transfer learning model. 

```bash
select transfer learning model: 
1.ResNet50V2 2.Xception 3.DenseNet201 4.MobileNetV2 :
ResNet50V2
```


### Contact Us
[Chun Yu Wu](https://github.com/dumplingman0403) - ericchunyuwu@gmail.com   
[Kao-Feng Hsieh](https://github.com/hsiehkaofeng) - hsiehkaofeng@gmail.com




