from PIL import Image
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from itertools import cycle
import matplotlib.pyplot as plt
import keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Embedding
from keras.layers import Conv2D, GlobalAveragePooling1D, MaxPooling2D
from keras.backend import tensorflow_backend as backend
import os
class StupidBrain:
    """
    This is the main class of this neural network, it's written by Bo Sun and Haixi Shan
    I have no idea why I use OOP in Python, maybe the reason is that I just feel it's better to get
    rid of some global variables
    """

    def __init__(self,train_data_path="",test_data_path="",seg_data_path="",csv_data_path = "",scale="RGB"):
        """
        This is the initializing method for StupidBrain, which accept some setting

        :param train_data_path: The path of training data folder
        :param test_data_path:  The path of test data folder
        :param seg_data_path:   The path of seg data folder
        :param scale:  The scale of the picture when training the model
        """
        if train_data_path != "" and not train_data_path.endswith("/"):
            train_data_path += "/"
        if test_data_path != "" and not test_data_path.endswith("/"):
            test_data_path += "/"
        if seg_data_path != "" and not seg_data_path.endswith("/"):
            seg_data_path += "/"
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.seg_data_path = seg_data_path
        self.csv_data_path = csv_data_path
        self.scale = scale
        self.test_data_prepared = False
        #self.input_dim = 750000 if self.scale == "RGB" else 250000

    class _Util:
        """
        This is the Utilize class which contains some useful method
        It's private and can only be used in StupidBrain
        """
        @classmethod
        def read_img(cls, path: str, img_name, *, scale='RGB', isFlatten = False):
            """
            This method is used to convert the picture to array, which is used frequently
            :param path: The path of the picture whose name is not included in the path.
            :param img_name:  The name of the picture, normally it's given by the csv file
            :param scale:  The scale of the picture, RGB is the default setting
            :param isFlatten: If change the array of the picture to 1d array
            :return: The array of the picture
            """
            if path != "" and not path.endswith('/'):
                path += "/"
            if not img_name.endswith('.png'):
                img_name += '.png'
            try:
                img = Image.open(path + img_name).convert(scale)
            except:
                raise Exception("Can't find the picture in this folder: ",path+img_name)
            if isFlatten:
                result = np.array(img).reshape(1, (3 if scale == 'RGB' else 1) * img.size[0] * img.size[1])
            else:
                image_region = (100,100,400,400)
                img = img.resize((180,180),box=image_region)
                result = np.array(img)
            
            return result

        @classmethod
        def read_csv(cls,file):
            """
            This is the method mainly used to read the csv file
            :param file: The file path
            :return: The whole data in the csv file, which is a tuple, the first item is the name of the file without
            extension, the second item is the data used to train model
            """
            csv = pd.read_csv(file,header=None)
            return csv.iloc[:, 0].values.astype('str'), csv.iloc[:, 1:].values.astype('int16')

        @classmethod
        def remove_background(cls, background_img_full, seg_img_full, scale, *, need_output=False, output_folder="./output", to_flatten = False,**kwargs):
            """
            This method is used to remove the background for question 2
            :param background_img_full: The path of the background image
            :param seg_img_full: The path of the seg image
            :param scale: The scale of the picture
            :param need_output: If the result picture should be saved
            :param output_folder: If saved, the output folder, default is a folder named output in current work directory
            :return: The array of the background removed picture
            """
            img_with_bg = cls.read_img("", background_img_full, scale=scale, isFlatten=False)
            img_with_seg = cls.read_img("", seg_img_full, scale=scale, isFlatten=False)
            if need_output:
                i = Image.fromarray(img_with_seg//255*img_with_bg)
                if output_folder != "" and not output_folder.endswith("/"):
                    output_folder+="/"
                i.save(output_folder + background_img_full.split("/")[-1] + ".png")
            if to_flatten:
                return (img_with_seg//255*img_with_bg).reshape([1,-1])
            return img_with_seg//255*img_with_bg

        @classmethod
        def contain(cls,folder,file_name):
            """
            This is the method to test if a file is in a certain folder
            :param folder: The folder
            :param file_name: The file name
            :return: A boolean value to show if the file is in the folder
            """
            if folder != "" and not folder.endswith("/"):
                folder += "/"
            files = glob.glob(folder+"*.png")
            return file_name in files

    def prepare_data(self,merge = False,*,questions = ("0","1","2"),question="0",shuffle = True,use_mask = True,prepare_test_data = False,batch_size = 32,test_ratio = 0.1,**kwargs):
        """
        After execution, the generator will be created for further use.
        :param merge: Merge the questions into one model
        :param questions: Questions to be merged
        :param question: If don't merge, which question would be trained
        :param shuffle: If shuffle the data
        :param use_mask: If use the mask of question 2
        :param prepare_test_data: If prepare the test data.
        :param kwargs: Other
        :return: None
        """
        self.prepare_test_data = prepare_test_data
        def inner_prepare_data():

            # The following code is used to process the training data
            file_names, outputs = self._Util.read_csv(self.csv_data_path) # Read the csv file
            file_names = file_names.reshape(-1, 1)
            temp_for_shuffle = np.concatenate((file_names, outputs), axis=1)
            print(file_names.shape)
            if shuffle: # If the data need be shuffled, then shuffle the data
                np.random.shuffle(temp_for_shuffle)

            file_names, outputs = temp_for_shuffle[:, 0], temp_for_shuffle[:, 1:]

            data_names = [] # Used to hold all the files used to train the model
            lables = [] # Corresponding labels
            if merge: # If merge the questions
                for i,file_name in enumerate(file_names):
                    if file_name.split("_")[0] in questions:
                        data_names.append(file_name)
                        lables.append(outputs[i])

            else:
                for i, file_name in enumerate(file_names):
                    if file_name.split("_")[0] == str(question):
                        data_names.append(file_name)
                        lables.append(outputs[i])

            if prepare_test_data and not self.test_data_prepared:
                print("Test data preparing...")
                temp_for_choose_test = np.concatenate((np.array(data_names).reshape(-1,1), np.array(lables).reshape(-1,4)), axis=1)
                idx = np.random.randint(temp_for_choose_test.shape[0], size=int(len(temp_for_choose_test)*test_ratio))
                test_data_full = temp_for_choose_test[idx, :]
                temp_for_choose_test = np.delete(temp_for_choose_test,idx,0)
                data_names, lables = temp_for_choose_test[:,0].tolist(),temp_for_choose_test[:,1:]
                test_data_X,test_data_Y = test_data_full[:,0],test_data_full[:,1:]
                self._process_test_data(test_data_X,test_data_Y,use_mask)
                self.test_data_prepared =True

            data_output = []
            label_output = []
            self.steps_per_epoch = len(data_names)//batch_size+1 
            for i, data_name in cycle(enumerate(data_names)):
                if use_mask and data_name.split("_")[0]=="2": # Consider question 2
                    img_bg = self.train_data_path+data_name+".png"
                    img_seg = self.seg_data_path+"0"+data_name.split("_")[-1]+".png"
                    img = self._Util.remove_background(img_bg,img_seg,scale=self.scale,**kwargs)
                else:
                    img = self._Util.read_img(self.train_data_path,data_name,scale=self.scale)
                if i % (batch_size+1) != batch_size:
                    data_output.append(img)
                    label_output.append(lables[i].reshape(4).astype('int16'))
                else:
                    yield np.array(data_output),np.array(label_output)
                    data_output.clear()
                    label_output.clear()
        next(inner_prepare_data()) # Run the generator once, and then initialize the steps_per_epoch
        self.generator = inner_prepare_data()

    def _process_test_data(self,X : list,Y : list,use_mask):
        x = []
        y = []
        for i, data_name in enumerate(X):
            if use_mask and data_name.split("_")[0] == "2":  # Consider question 2
                img_bg = self.train_data_path + data_name + ".png"
                img_seg = self.seg_data_path + "0" + data_name.split("_")[-1] + ".png"
                img = self._Util.remove_background(img_bg, img_seg, scale=self.scale)
            else:
                img = self._Util.read_img(self.train_data_path, data_name, scale=self.scale)
            x.append(img)
            y.append(Y[i].reshape(4).astype('int16'))
        self.test_X = np.array(x)
        self.test_Y = np.array(y)

        
    def rmse(self,true_value,pred_value):
        return backend.sqrt(backend.mean(backend.square(true_value-pred_value),axis=-1))
    save_path = './weights.hdf5'
    callback_list = [
        keras.callbacks.ModelCheckpoint(save_path,monitor='val_loss',verbose=0,
                                        save_best_only=True,save_weights_only=True,mode='auto',period=1),
        keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1,patience=5,verbose=0,mode='min'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=3,verbose=1,mode='auto')]

    def run_model(self,*,model=None,steps_per_epoch=500,epochs=5,test_data=None):
        """
        Run the model with limited setting
        :param model: The model can be created by user or by default
        :param steps_per_epoch: Ya
        :param epochs: Ya
        :return: None
        """

        with tf.device('/device:GPU:0'):
            if model==None:
                model = Sequential()
                model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(180, 180, 3)))
                model.add(Conv2D(32, (5, 5), activation='relu'))
                model.add(MaxPooling2D(pool_size=(3, 3)))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Flatten())
                model.add(Dense(256, activation='relu'))
                model.add(Dense(4, activation='linear'))
                #rms = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.001)
                adam = keras.optimizers.Adam(0.002,0.9,0.999)   
                #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(loss='mse', optimizer=adam,metrics=[self.rmse])
                #model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
            print(self.test_X.shape)
            print(self.test_Y.shape)
            model.summary()
            model.save_weights(self.save_path)
            result = model.fit_generator(self.generator,samples_per_epoch=self.steps_per_epoch,epochs=epochs,
                                    validation_data=(self.test_X,self.test_Y),callbacks = self.callback_list)
            print(result.history.keys())
            plt.plot(result.history['rmse'])
            plt.plot(result.history['val_rmse'])
            plt.title('rmse of training set and validation set')
            plt.ylabel('rmse')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            if test_data:
                results = model.predict(self.test_X,self.test_Y)
                return results
            #if self.prepare_test_data:
              #  print(model.evaluate(self.test_X,self.test_Y))

    def save_toCSV(pred_data,level):
        data = data.astype('int64')
        data_holder = pd.DataFrame(data=pred_data)
        if level == '0':
            data_holder = np.column_stack([level0_image,pred_data])
        elif level == '1':
            data_holder = np.column_stack([level1_image,pred_data])
        elif level == '2':
            data_holder = np.column_stack([level2_image,pred_data])
        data_holder.to_csv('./no1'+level+'.csv',index=False,header=csv_header)
        print(data_holder)

    level0_image = []
    level1_image = []
    level2_image = []

    def pre_test_data(self):
        csv_header = ['id','x1','y1','x2','y2']
        #folder_path = 'F:/test/test/'
        image_list = os.listdir(self.test_data_path)
        for i in range(len(image_list)):
            if image_list[i].split('_')[0] == '0':
                self.level0_image.append(image_list[i])
            elif image_list[i].split('_')[0] == '1':
                self.level1_image.append(image_list[i])
            elif image_list[i].split('_')[0] == '2':
                self.level2_image.append(image_list[i])

if __name__ == '__main__':
    s = StupidBrain(train_data_path="../input/train/train/",test_data_path="../input/test/test/",seg_data_path="../input/seg/seg/",csv_data_path="../input/train/train/annotations.csv")
    s.pre_test_data()
    s.prepare_data(True,questions = ("1",'2'),prepare_test_data = True)
    #s.run_model(epochs=40)
    results = s.run_model(epochs=1,test_data=s.level1_image)
    s.save_toCSV(results,'0')