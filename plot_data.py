import matplotlib.pyplot as plt
import pandas as pd
import os


PATH = 'Images-processed'
CATEGORY = ['CT_COVID', 'CT_NonCOVID']

CAT_list = []
for c in CATEGORY:
    folder_path = os.path.join(PATH, c)    
    file_list = os.listdir(folder_path)
    for f in file_list:
        if f != '.DS_Store':
            CAT_list.append(c)


plt.hist(CAT_list, bins = 5 ,stacked=True, label='123')

plt.show()
