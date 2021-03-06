from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import cycle

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
        self.input_dim = 750000 if self.scale == "RGB" else 250000

    class _Util:
        """
        This is the Utilize class which contains some useful method
        It's private and can only be used in StupidBrain
        """
        @classmethod
        def read_img(cls, path: str, img_name, *, scale='RGB', isFlatten = True):
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
                result = np.array(img).reshape(-1)
            else:
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
        def remove_background(cls, background_img_full, seg_img_full, scale, *, need_output=False, output_folder="./output", to_flatten = True,**kwargs):
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
                return (img_with_seg//255*img_with_bg).reshape(-1)
            return img_with_seg//255*img_with_bg


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
                    data_output.append(img.reshape(500,500,3))
                    label_output.append(lables[i].reshape(4,1,1).astype('int16'))
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



    def run_model(self,*,model=None,steps_per_epoch=500,epochs=5):
        """
        Run the model with limited setting
        :param model: The model can be created by user or by default
        :param steps_per_epoch: Ya
        :param epochs: Ya
        :return: None
        """

        with tf.device('/device:GPU:0'):
            if model==None:
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Dense(128, input_dim=self.input_dim, activation='relu', kernel_initializer='random_uniform',
                                          bias_initializer='zeros'))
                model.add(tf.keras.layers.Dropout(0.2))
                model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform',
                                          bias_initializer='zeros'))
                model.add(tf.keras.layers.Dense(4, activation='linear', kernel_initializer='random_uniform',
                                          bias_initializer='zeros'))
                rms = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.001)
                model.compile(optimizer=rms, loss=tf.keras.losses.mean_absolute_error,metrics=['accuracy'])
            # model.summary()
            model.fit_generator(self.generator,steps_per_epoch=self.steps_per_epoch,epochs=epochs)
            if self.prepare_test_data:
                print(model.evaluate(self.test_X,self.test_Y))


if __name__ == '__main__':
    s = StupidBrain(train_data_path="/Users/sunbo/Downloads/all/train",test_data_path="/Users/sunbo/Downloads/all/test",seg_data_path="/Users/sunbo/Downloads/all/seg",csv_data_path="/Users/sunbo/Downloads/all/train/annotations.csv")
    s.prepare_data(True,questions = ("1","2"))
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3),data_format="channels_last"))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='linear', kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    s.run_model(model=model,epochs=15)