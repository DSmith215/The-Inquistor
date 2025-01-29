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
        y = Dense(1, activation="sigmoid")(y)

        return Model(input = x, output = y)

#Instantiate a MesoNet model with pretrained weights
meso = Meso4()
meso.load('./weights/Meso_DF')

#Prepae image data

#Rescaling pixel values (between 1 and 255) to a range between 0 and 1
Data_Generator = ImageDataGenerator(rescale = 1./255)

#Instantiating generator to feed images through the network
Generator = Data_Generator.flow_from_directory(
    './data/',
    target_size = (256, 256),
    batch_size = 1, 
    class_mode = 'binary')

#Check class assignment
Generator.class_indices

#Removing potential autosave file, should this be run in a jupyter notebook
!rmdir /s /q c:data\.ipynb_checkpoints

#Rendering image X with label y for MesoNet
X, y = Generator.next()

#Evaluating prediction
print(f"Predict likelihood: {meso.predict(X)[0][0]:.4f}")
print(f"Actual label: {int(y[0])}")
print(f"\nCorrect prediction: {round(meso.predict(X)[0][0]==y[0]}")

#Display the image we're testing
plt.imshow(np.squeeze(X));

#Creating seperate lists for correctly classified and missclassified images

#Correct lists
#Real images
Correct_Real = []
Correct_Real_Prediction = []

#Deepfaked images
Correct_Deepfake = []
Correct_Deepfake_Predictions = []

#Incorrect lists
#Real images
Misclassified_Real = []
Misclassified_Real_Prediction = []

#Deepfaked images
Misclassified_Deepfake = []
Misclassified_Deepfake_Prediction = []
