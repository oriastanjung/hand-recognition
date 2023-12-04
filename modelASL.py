import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# save dan run

# sekrang kita buat path ke kaggle dataset dlu
folderData ="kaggleData"
files = os.listdir(folderData)

# list dari files yang ada di folder
# sort dari A-Y

files.sort()

# kita coba print files2 nya ke terminal
print(files) # ini akan cetak Folder A-Y 


# buat list dari setiap gambar berdasarkan labelnya

image_array = []
label_array = []

# looping smua folder yang ada di folderData

for i in range(len(files)):
    # list dari setiap gambar yang ada pada folder 
    sub_file=os.listdir(folderData+"/"+files[i])

    # check jumlah file setiap folder
    # print(len(sub_file))

    # looping semua sub file
    for j in range(len(sub_file)):
        # ini path setiap gambarnya
        #       kaggleData/A/image_name1.jpg
        file_path = folderData + "/" + files[i] + "/" + sub_file[j]
        
        # baca setiap gambar
        image=cv2.imread(file_path)

        # kita resize gambar menjadi 96x96
        image=cv2.resize(image, (96,96))

        # convert gambar BGR ke RGB 
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # add gambar ini ke image_array
        image_array.append(image)

        # tambahkan juga labelnya
        # i itu index dari 0 sampai jumlah folder atau label yg ada
        # jadi ini akan menjadi label nya
        label_array.append(i)
        # di tutorial menit ke 14.25



