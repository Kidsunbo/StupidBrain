from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection
from itertools import cycle


'''
Start global variable
'''
scale = "RGB"
isFlatten = True
input_dim = 750000 if scale=="RGB" else 250000
questions = ("0",)
batch_size = 32

'''
End global variable
'''

'''
Start util area
'''


def read_img(path: str, img_name, *, scale='RGB', isFlatten=isFlatten):

    if path != "" and not path.endswith('/'):
        path += "/"
    if not img_name.endswith('.png'):
        img_name += '.png'
    try:
        img = Image.open(path + img_name).convert(scale)
    except:
        raise Exception("Can't find the picture in this folder: ", path + img_name)
    if isFlatten:
        result = np.array(img).reshape(-1)
    else:
        result = np.array(img)
    return result


def read_csv(file):
    csv = pd.read_csv(file, header=None)
    result = csv.values
    np.random.shuffle(result)
    return result[:,0].astype('str'), result[:,1:].astype('int16')


def remove_background(background_img_full, seg_img_full, scale, *, need_output=False, output_folder="./output",
                      to_flatten=True):
    img_with_bg = read_img("", background_img_full, scale=scale, isFlatten=False)
    img_with_seg = read_img("", seg_img_full, scale=scale, isFlatten=False)
    if need_output:
        i = Image.fromarray(img_with_seg // 255 * img_with_bg)
        if output_folder != "" and not output_folder.endswith("/"):
            output_folder += "/"
        i.save(output_folder + background_img_full.split("/")[-1] + ".png")
    if to_flatten:
        return (img_with_seg // 255 * img_with_bg).reshape(-1)
    return img_with_seg // 255 * img_with_bg


'''
End util area
'''


def get_data(all_folder,questions = questions):
    global train_x
    global train_y
    global test_x
    global test_y

    img_name_all, labels_all = read_csv(all_folder+"train/annotations.csv")
    img_data = []
    label_data = []
    for i, name in enumerate(img_name_all):
        if name.split("_")[0] in questions:
            if name.split("_")[0] != "2":
                img_data.append(read_img(all_folder+"/train/",name+".png",scale=scale,isFlatten=isFlatten))
            else:
                img_data.append(remove_background(
                    background_img_full=all_folder+"/train/"+name+",png",
                    seg_img_full=all_folder+"/seg/"+"0"+name+".png",
                    scale=scale,to_flatten=isFlatten
                ))
            label_data.append(labels_all[i])

    train_x, test_x, train_y,test_y = sklearn.model_selection.train_test_split(np.array(img_data),np.array(label_data),test_size=0.1)


def get_data_generator(all_folder):
    global test_x
    global test_y
    global steps_per_epoch

    img_name_all, labels_all = read_csv(all_folder+"train/annotations.csv")
    img_names = []
    labels = []
    for i, name in enumerate(img_name_all):
        if name.split("_")[0] in questions:
                img_names.append(name)
                labels.append(labels_all[i])

    train_x_name, test_x_name, train_y_label, test_y_label = sklearn.model_selection.train_test_split(img_names,
                                                                                        labels,
                                                                                        test_size=0.1)
    print(len(train_x_name),len(test_x_name))
    test_x_temp = []
    test_y_temp = []
    print("Prepare test data...")
    for i, test_name in enumerate(test_x_name):
        if test_name.split("_")[0] != "2":
            test_x_temp.append(read_img(all_folder + "/train/", test_name + ".png", scale=scale, isFlatten=isFlatten))
        else:
            test_y_temp.append(remove_background(
                background_img_full=all_folder + "/train/" + test_name + ",png",
                seg_img_full=all_folder + "/seg/" + "0" + test_name + ".png",
                scale=scale, to_flatten=isFlatten
            ))
        test_y_temp.append(test_y_label[i])
    test_x = np.array(test_x_temp)
    test_y = np.array(test_y_temp)

    print("Prepare train data...")
    while True:
        img_output = []
        labels_output = []
        steps_per_epoch = len(train_x_name)//batch_size+1
        for i, name in enumerate(train_x_name):
            if name.split("_")[0] != "2":
                img_output.append(read_img(all_folder+"/train/",name+".png",scale=scale,isFlatten=isFlatten))
            else:
                img_output.append(remove_background(
                    background_img_full=all_folder+"/train/"+name+",png",
                    seg_img_full=all_folder+"/seg/"+"0"+name+".png",
                    scale=scale,to_flatten=isFlatten
                ))
            labels_output.append(train_y_label[i])
            if i%(batch_size+1) == batch_size:
                yield np.array(img_output),np.array(labels_output)
                img_output = []
                labels_output=[]
        yield np.array(img_output), np.array(labels_output)



def build_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(128, input_dim=input_dim, activation='relu', kernel_initializer='random_uniform',
                              bias_initializer='zeros'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform',
                                    bias_initializer='zeros'))
    model.add(tf.keras.layers.Dense(4, activation='linear', kernel_initializer='random_uniform',
                                    bias_initializer='zeros'))
    rms = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.001)
    model.compile(optimizer=rms, loss=tf.keras.losses.mean_absolute_error)
    model.summary()
    return model


def build_cnn_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3), data_format="channels_last"))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(
        tf.keras.layers.Dense(4, activation='linear', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model


def write_to_csv(X,result):
    pass

if __name__ == '__main__':

    # build_cnn_model().fit(train_x,train_y,batch_size=32,validation_data=(test_x,test_y),epochs=3)
    # next(get_data_generator("/Users/sunbo/Downloads/all/"))
    # get_data("../input/train/")

    next(get_data_generator("/Users/sunbo/Downloads/all/"))

    print(len(train_x),len(train_y),len(test_x),len(test_y))
    # build_model().fit(train_x,train_y,validation_data=(test_x, test_y), epochs=3)