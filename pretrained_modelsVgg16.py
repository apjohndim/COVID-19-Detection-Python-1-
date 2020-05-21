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
from keras.applications.vgg19 import VGG19
from keras.utils import plot_model



#%%   


def make_model():
    
    print("[INFO] Compiling Model...")
    
    base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(400, 300, 3), pooling=None, classes=3)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[18:]:
        layer.trainable = True
    #base_model.summary()
    
    x = layer_dict['block5_conv3'].output
    x= GlobalAveragePooling2D()(x)
    
    x = Dense(1024, activation='relu')(x)
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
imagePaths = sorted(list(paths.list_images('C:\\Users\\User\\COVID19 DETECTION\\dataset normal')))  
random.seed(SEED)
random.shuffle(imagePaths)


# loop over the input images
for imagePath in imagePaths:
    
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (300, 400))/255
    data.append(image)
    label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float")
labels = np.array(labels)
Labels_verbal = labels

print("[INFO] Private data images loaded!")

print("Reshaping data!")

#data = data.reshape(data.shape[0], 400, 300, 3)


print("Data Reshaped to feed into models channels last")

from sklearn.preprocessing import MultiLabelBinarizer
print("Labels formatting")
lb = MultiLabelBinarizer()
labels = lb.fit_transform(labels) 
print("Labels ok!")


#%%
time1 = time.time() #initiate time counter
n_split=10 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
name2 = 5000 #name initiator for the incorrectly classified insatnces

omega = 1


for train_index,test_index in KFold(n_split).split(data):
    trainX,testX=data[train_index],data[test_index]
    trainY,testY=labels[train_index],labels[test_index]

    model3 = make_model() #in every iteration we retrain the model from the start and not from where it stopped
    if omega == 1:
       model3.summary()
    omega = omega + 1   
    
    print('[INFO] PREPARING FOLD: '+str(omega-1))
    model3.fit(trainX, trainY,epochs=5, batch_size=64)
    score = model3.evaluate(testX,testY)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',model3.evaluate(testX,testY))
    
    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    testY = testY.argmax(axis=-1)
    test_labels = np.concatenate ([test_labels, testY]) #merge the two np arrays of labels
#scores = np.asarray(scores)
#final_score = np.mean(scores)


print("[INFO] Results Obtained!")
print('Time taken: {:.1f} seconds'.format(time.time() - time1)) 



