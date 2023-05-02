from myLayers import *
from dataHandle import *
from datetime import datetime

config = config_read_in()
readIn_img_size = (config['readIn_img']['width'],
                   config['readIn_img']['height'],
                   config['readIn_img']['dim'])
model_name = config['model_name']


def model4_5(input_shape=readIn_img_size):
    inp = Input(shape=input_shape)

    conv1 = conv_batch_act(channels=96)(inp)
    max_pool1 = MP2D((2, 2))(conv1)
    conv2 = conv_batch_act(channels=96)(max_pool1)
    max_pool2 = MP2D((2, 2))(conv2)
    conv3 = conv_batch_act(channels=128)(max_pool2)
    max_pool3 = MP2D((2, 2))(conv3)
    conv4 = conv_batch_act(channels=256)(max_pool3)
    max_pool4 = MP2D((2, 2))(conv4)

    de_conv1 = de_conv_comb(channels=256)(max_pool4, conv4)  # 12*12*256
    de_conv2 = de_conv_comb(channels=256, de_padding='valid')(de_conv1, conv3)  # 25*25*256

    # -----------------categorical and LR branch start----------------------
    cate_conv1 = conv_batch_act(channels=128)(de_conv2)
    cate_conv2 = conv_batch_act(channels=64)(cate_conv1)

    glob_gap = GlobalAP()(cate_conv2)  # global pooling
    _LR = Dense(1, activation='sigmoid', name="LR_output")(glob_gap)  # LR result

    cate_dense = Dense(128, activation='relu')(glob_gap)
    _cate = Dense(10, activation='sigmoid', name="category_output")(cate_dense)
    # ------------------categorical and LR branch end-----------------------

    # ------------------finger map branch start----------------------
    fm_conv1 = conv_batch_act(channels=256)(de_conv2)
    fm_conv2 = conv_batch_act(channels=256)(fm_conv1)
    fm_conv3 = conv_batch_act(channels=256)(fm_conv2)
    fm_conv4 = conv_batch_act(channels=256)(fm_conv3)

    de_conv3 = de_conv_comb(channels=256, combine='concatenate')(fm_conv4, conv2)

    fm_conv5 = conv_batch_act(channels=128)(de_conv3)

    _fm = Conv2D(6, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding="same",
                 name="heatmap_output")(fm_conv5)
    # -------------------finger map branch end---------------------------

    model_structure = Model(inputs=inp, outputs=[_LR, _cate, _fm])
    return model_structure


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    # model = model4_5()
    model = eval(model_name+'()')
    # model.summary()
    now = datetime.now()
    save = True
    if save:
        if save_model_structure(model, now):
            print("Structure save successful")
        else:
            print("Structure save failed!!!!!")
