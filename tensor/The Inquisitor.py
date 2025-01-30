#Import packages
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow

#Height and Width refer to the size of the image
#Channels refers to the amount of color channels (red, green, blue)

#Image dimensions function
im_dim = {"height": 256, "width": 256, "channels": 3}

#Create a classifier class

class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        return self.model.pred(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

#Creata MesoNet class using Classifier

class Meso(Classifier):
    def __init__(slef, learning_rate = 0.001):
        self.model = self.init.model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer,
                           loss = "mean_squared_error",
                           netrics = ['accuracy'])
    
    def init_model(self):
        x = Input(shape = (im_dim['height'],
                           im_dim['width'],
                           im_dim['channel']))
        
        x1 = Conv2D(8, (3,3), padding = 'same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2,2), padding = 'same')(x1)

        x2 = Conv2D(8, (5,5), padding = 'same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2,2), padding = 'same')(x2)

        x3 = Conv2D(16, (5,5), padding = 'same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2,2), padding = 'same')(x3)

        x4 = Conv2D(8, (3,3), padding = 'same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(2,2), padding = 'same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha = 0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activaton = "sigmoid")(y)

        return Model(inputs = x, outputs = y)

#Instantiate a MesoNet model with pretrained weights
meso = Meso4()
meso.load('./weights/Meso4_DF')

#Prepare the image we're classifing

#Rescaling pixel values (between 1 and 255) to a range between 0 and 1
Data_Generator = Image_Data_Generator(rescale = 1.0/255)

#Instantiate the generator to feed the network the images we're trying to classify
Generator = Data_Generator.flow_from_directory(
    './data/',
    target_size = (256,256),
    batch_size = 1,
    class_mode = 'binary')

#Check the class assignments
Generator.class_indices

#Remove hidden files from Jupyter files
!rmdir /s /q c:data\.ipynb_checkpoints
