import os
from tqdm import tqdm
import time
from PIL import Image
import uuid
# function untuk save gambar dengan random name
def saveImageWithTimeStamp(image, label, folder):
    
    currentTime = int(time.time())
     # Generate a random UUID
    random_suffix = str(uuid.uuid4().hex)[:8]
    filename = f"{label}_{currentTime}_{random_suffix}.jpg"
    filepath = os.path.join(folder, filename)
    # Check if the directory exists, create it if not
    os.makedirs(folder, exist_ok=True)

    # Save using PIL
    image.save(filepath)
    print("Saved to " + filepath)


# operasi sobel untuk deteksi tepi
def operatorSobel(image):
    # rubah ke graysacle dahulu
    img_gray = image.convert('L')

    # dapatin size width dan height
    width, height = img_gray.size
    # sobel kernel
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # buat arrray hasil perhitungan sobel
    sobel_result = [[0] * width for _ in range(height)]
    
    # terapin persamaan sobel
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            pixel_x = sum(img_gray.getpixel((min(width-1, max(0, x + i)), min(height-1, max(0, y + j)))) * sobel_x[i][j] for i in range(3) for j in range(3))
            pixel_y = sum(img_gray.getpixel((min(width-1, max(0, x + i)), min(height-1, max(0, y + j)))) * sobel_y[i][j] for i in range(3) for j in range(3))
            sobel_result[y][x] = int((pixel_x**2 + pixel_y**2)**0.5)
    
    # buat gambar dari sobel_result
    sobel_image = Image.new('L', (width, height))
    sobel_image.putdata([pixel for row in sobel_result for pixel in row])
    
    return sobel_image

# save dan run

# sekrang kita buat path ke kaggle dataset dlu
folderData ="kaggleData"
outputFolder = "dataset"
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

for i in tqdm(range(len(files))):
    # list dari setiap gambar yang ada pada folder 
    sub_file=os.listdir(folderData+"/"+files[i])


    # looping semua sub file
    for j in range(len(sub_file)):
        # ini path setiap gambarnya
        #       kaggleData/A/image_name1.jpg
        file_path = folderData + "/" + files[i] + "/" + sub_file[j]
        
        img=Image.open(file_path)

        img= img.resize((256,256))

        # terapin operator sobel
        sobelImage=operatorSobel(img)



        # simpan gambar hasil sobel
        saveImageWithTimeStamp(sobelImage,files[i],os.path.join(outputFolder,files[i]))
        




