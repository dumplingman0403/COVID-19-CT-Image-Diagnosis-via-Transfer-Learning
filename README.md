# COVID-19-CT-Image-Diagnosis-via-Tranfer-Learning

## Introduction

## How to run

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
    batch_size = 8                                  
    epochs = 40                                     # Number of epochs
    ntrain = 0.8 * len(X_train)                     # split data with 80/20 train/validation 
    nval = 0.2 * len(X_train)

    X_train, X_val, y_train, y_val = train_test_split(  # 20% validation set
        x, y_train, test_size=0.20, random_state=2)
```


```bash
python3 Model.py
```



