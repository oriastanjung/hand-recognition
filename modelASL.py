import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# sekrang kita buat path ke kaggle dataset dlu
folderData ="dataset"
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
# tqdm untuk ada loading di console terminal
for i in tqdm(range(len(files))):
    # list dari setiap gambar yang ada pada folder 
    sub_file=os.listdir(folderData+"/"+files[i])

    # check jumlah file setiap folder
    # print(len(sub_file))

    # looping semua sub file
    for j in range(len(sub_file)):
        # ini path setiap gambarnya
        #       dataset/A/image_name1.jpg
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

# kita convert list ke array

image_array=np.array(image_array)
label_array=np.array(label_array,dtype="float")

# melakukan split dataset menjadi data train dan data test

from sklearn.model_selection import train_test_split
# ambil 15% dataset untuk testing
# output                                            train image     label       spliting size
X_train, X_test, Y_train, Y_test = train_test_split(image_array, label_array, test_size=0.15)

del image_array, label_array

# untuk membebaskan ram
import gc
gc.collect()

# X_train akan menggunakan 85% dari dataset gambar
# X_test hanya menggunakan 15% dari dataset gambar

# membuat model CNN

from keras import layers, callbacks, utils, applications, optimizers
from keras.models import Sequential, Model, load_model

model=Sequential()

# kita akan menambah pre trained model CNN bernama EfficientNetB0

pretrainedModel=tf.keras.applications.EfficientNetB0(input_shape=(96,96,3), include_top=False)
model.add(pretrainedModel)

# tambah Polling ke model 
model.add(layers.GlobalAveragePooling2D())

# tambah dropout ke model
# dropout ini untuk meningkatkan akurasi dengan mengurangi overfitting

model.add(layers.Dropout(0.3))

# lalu kita tambahkan dense layer sebagai output

model.add(layers.Dense(1))

# untuk beberapa versi tensorflow kita memerlukan untuk melakukan build model

model.build(input_shape=(None,96,96,3))


# untuk melihat rangkuman model

model.summary()


# save dan jalankan untuk melihat model summary
# pastikan menggunakan internet untuk download pretrained modelnya
# Kita gunakan GPU untuk train model agar lebih cepat


# compile model CNN
# disini menggunakan optimizer dan loss untuk meningkatkan akurasi
model.compile(optimizer="adam",loss="mae",metrics=["mae"])

# buat checkpoint untuk menyimpan best accuracy model

ckp_path="trained_model/model"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,monitor="val_mae", mode="auto", save_best_only=True, save_weights_only=True)

# monitor : monitor memvalidasi mae loss untuk menyimpan model
# mode : digunakan untuk memilih menyimpan val_mae ketika minimum atau maximum
# ini memiliki 3 option : "min", "max", "auto"
# ketika val_mae berkurang model akan tersimpan
# save_best_only : False -> akan menyimpan seluruh model
# save_weights : simpan hanya beban nya

# buat tingkat pembelajaran atau learning rate reducer untuk berkurkan 1r ketika akurasi tidak berkembang

reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(factor=0.9, monitor="val_mae", mode="auto", cooldown=0, patience=5, verbose=1, min_lr=1e-6)

# factor : ketika berkurang, next lr akan menjadi 0.9 dari nilai sekarang
# patience=X
# kurangi lr setelah X epoch ketika akurasi tidak berkembang
# verbose : menampilkan setelah setiap epoch
# min_lr : minimum learning rate

# Start training model

Epoch=100
Batch_Size=32

# Pilih size batch nya tergantung gpu card

# X_train, X_test, Y_train, Y_test
history=model.fit(X_train, Y_train, validation_data=(X_test,Y_test), batch_size=Batch_Size,epochs=Epoch, callbacks=[model_checkpoint, reduce_lr])

# setelah training selesai kita load best model

model.load_weights(ckp_path)

# convert model ke tensorflow lite model

converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

# save model

with open("model.tflite","wb") as f:
    f.write(tflite_model)

# untuk menampilkan hasil prediksi ke test dataset

prediction_val = model.predict(X_test, batch_size=32)

# print 10 nilai pertama
print(prediction_val[:10])
# print 10 nilai pertama dari Y_test
print(Y_test[:10])

#loss: 3.2265 - mae: 3.2265 - val_loss: 2.3238 - val_mae: 2.3238 - lr: 0.0010
# jika val_mae berkurang maka model semakin berkembang
# loss: 1.4266 - mae: 1.4266 - val_loss: 1.4932 - val_mae: 1.4932 - lr: 0.0010
