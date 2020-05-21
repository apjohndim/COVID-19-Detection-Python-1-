print("[INFO] Importing Libraries")
import matplotlib as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# matplotlib inline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import time   # time1 = time.time(); print('Time taken: {:.1f} seconds'.format(time.time() - time1))
import warnings
import keras
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from PIL import Image 
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time
from sklearn.metrics import classification_report, confusion_matrix
from keras_applications.resnet import ResNet50
from keras_applications.mobilenet import MobileNet
SEED = 50   # set random seed
print("[INFO] Libraries Imported")
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model


#%%   


def make_model():

    print("[INFO] Compiling Model...")
    
    #base_model = keras.applications.resnet_v2.ResNet152V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=3)
    base_model = keras.applications.mobilenet.MobileNet(input_shape=(266,200,3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=3)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    
    # #for layer in base_model.layers:
    #     #print (layer.name)
    
    
    # plot_model(base_model, to_file='basevgg16.png')
    for layer in base_model.layers:
          layer.trainable = True
    #for layer in base_model.layers[730:]: #block8_7_ac
          #layer.trainable = True
    # #base_model.summary()
    
    x = layer_dict['conv_pw_13_relu'].output
    x= GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(750, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    
    x = Dense(3, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)
    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod1.png')
    print("[INFO] Model Compiled!")
    return model
  
#%%
  
print("[INFO] loading images from private data...")
data = []
labels = []
labels2 = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('C:\\Users\\User\\COVID19 DETECTION\\dset_3 class')))  
random.seed(SEED)
random.shuffle(imagePaths)


# loop over the input images
for imagePath in imagePaths:
    
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (200, 266))/255
    data.append(image)
 
    label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(label)

data = np.array(data, dtype="float")
labels = np.array(labels)
Labels_verbal = labels



print("[INFO] Private data images loaded!")

print("Reshaping data!")
print("Data Reshaped to feed into models channels last")

from sklearn.preprocessing import MultiLabelBinarizer
print("Labels formatting")
lb = MultiLabelBinarizer()
labels = lb.fit_transform(labels) 
print("Labels ok!")


#%%
time1 = time.time() #initiate time counter
n_split=5 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept

omega = 1


for train_index,test_index in KFold(n_split).split(data):
    trainX,testX=data[train_index],data[test_index]
    trainY,testY=labels[train_index],labels[test_index]


    model3 = make_model() #in every iteration we retrain the model from the start and not from where it stopped
    if omega == 1:
       model3.summary()
    omega = omega + 1   
    
    print('[INFO] PREPARING FOLD: '+str(omega-1))
    model3.fit(trainX, trainY,epochs=4, batch_size=32)
    
    #aug = ImageDataGenerator(rotation_range=45, horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest')
    #aug.fit(trainX)
    #model3.fit_generator(aug.flow(trainX, trainY,batch_size=64), epochs=35, steps_per_epoch=len(trainX)//64)
    score = model3.evaluate(testX,testY)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',model3.evaluate(testX,testY))
    
    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    #predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    testY = testY.argmax(axis=-1)
    test_labels = np.concatenate ([test_labels, testY]) #merge the two np arrays of labels
#scores = np.asarray(scores)
#final_score = np.mean(scores)


print("[INFO] Results Obtained!")
print('Time taken: {:.1f} seconds'.format(time.time() - time1)) 




