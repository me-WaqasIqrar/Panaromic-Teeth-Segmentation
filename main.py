import os
import random
import matplotlib.pyplot as plt

from download_dataset import *
from images_prepare import *
from masks_prepare import *
from model import *
import torch


path = "./Dataset"

if not os.path.exists(path+'/DentalPanoramicXrays.zip'):
    os.makedirs(path, exist_ok=True)
    print("Started Downloading...")
    download_dataset(path)
else:
    print("Dataset already exists...")

#pre_images
X,X_sizes=pre_images((512,512),path,True)

Y=pre_splitted_masks(path='./Custom_Masks') #Custom Splitted MASKS size 512x512

X=np.float32(X/255)
Y=np.float32(Y/255)

x_train=X[:105,:,:,:]
y_train=Y[:105,:,:,:]
x_test=X[105:,:,:,:]
y_test=Y[105:,:,:,:]

random_number=random.randint(0,104)
print(random_number)
plt.imshow(x_train[random_number,:,:,0])
plt.show()
plt.imshow(y_train[random_number,:,:,0])
plt.show()
# Check if CUDA is available and set the device

model=UNET(input_shape=(512,512,1),last_activation='sigmoid')
model.summary()



model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Your choice batch and epoch
model.fit(x_train,y_train,batch_size=8,epochs=1,verbose=1)


print("Done...")

##Plotting - RESULT Example
img=cv2.imread("/content/Data/Images/107.png")#original img 107.png

predict1 = cv2.resize(predict, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

mask=np.uint8(predict1*255)#
_, mask = cv2.threshold(mask, thresh=255/2, maxval=255, type=cv2.THRESH_BINARY)
cnts,hieararch=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(img[:,:,0], cnts, -1, (255, 0, 0) , 2)
img = cv2.UMat.get(img)
cv2_imshow(img)