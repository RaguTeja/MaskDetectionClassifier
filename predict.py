from tensorflow.keras.preprocessing.image import load_img,img_to_array     # loading image from server
import numpy as np                                                         # resize the image as per the model

class Predict:

    def model_predict(self,file_path,model):                               # prediction functionality
        img=load_img(file_path,target_size=(224,224))                      # load image from server
        img=img_to_array(img)/255                                          # normalizing the image
        img=np.expand_dims(img,axis=0)                                     # increase the dimension
        prob=model.predict(img)                                            # prediction
        pred=np.argmax(prob)                                               # Finding class with more probability

        if pred==0:
            return 'THIS PERSON WORE MASK'
        else:
            return "THIS PERSON NOT WEARING MASK"

