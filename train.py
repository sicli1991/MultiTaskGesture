# import tensorflow as tf
from log import *
from model import *
from dataHandle import *
import os
from datetime import datetime
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
"""
logfile = 'temp_'+get_date_str()+'.txt'
log_folder = os.getcwd()+'/log'
os.chdir(log_folder)
rr_file = os.path.join(log_folder, logfile)
logger = WriteLog(rr_file)
"""
config = config_read_in()
dataset_path = config['dataset_path']
epochs = config['train_parameters']['epoch']
batch_size = config['train_parameters']['batch_size']
shuffle = config['train_parameters']['shuffle']


def loss_function_1(y_true, y_pred):
    """ Probabilistic output loss """
    a = tf.clip_by_value(y_pred, 1e-20, 1)
    b = tf.clip_by_value(tf.subtract(1.0, y_pred), 1e-20, 1)
    cross_entropy = - tf.multiply(y_true, tf.math.log(a)) - tf.multiply(tf.subtract(1.0, y_true), tf.math.log(b))
    cross_entropy = tf.reduce_mean(cross_entropy, 0)
    loss = tf.reduce_mean(cross_entropy)
    return loss


if __name__ == '__main__':
    now_time = datetime.now()
    train_X = os.path.join(dataset_path, 'Train_data', 'train_X.pickle')
    train_Y = os.path.join(dataset_path, 'Train_data', 'train_Y_combine_fw.pickle')
    train_Y_prob = os.path.join(dataset_path, 'Train_data', 'Train_fm_prob.pickle')
    train_Y_cate = os.path.join(dataset_path, 'Train_data', 'train_Y_cate.pickle')

    X = pickle_read_in(train_X)
    Y = pickle_read_in(train_Y)
    Y_prob = pickle_read_in(train_Y_prob)
    Y_cate = pickle_read_in(train_Y_cate)
    output_format = [Y_cate, Y_prob, Y, Y]  # output format defined here!!!

    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = eval(config['model_name']+"()")
    model.summary()
    opt = optimizer_generation(config["train_parameters"]["optimizer_used"])
    model.compile(loss={"category_output": "categorical_crossentropy", "mask_output": loss_function_1,
                        "fm_out_1": "mean_squared_error", "fm_out_2": "mean_squared_error"},
                  loss_weights=config['train_parameters']['loss_weights'],
                  optimizer=opt
                  )  # metrics=config['train_parameters']['metrics']
    # loss = config['train_parameters']['loss'],
    history = model.fit(X, output_format, epochs=epochs, batch_size=batch_size, shuffle=shuffle)

    """
    finalModel_json = model.to_json()
    with open("modelGen/finalModel_v5_3.json", "w") as json_file:
        json_file.write(finalModel_json)
    model.save_weights("modelGen/finalModel_v5_3.h5")
    print("model saved")
    """
    model_to_json(model, now_time)
    # checkpoint = ModelCheckpoint('model_check_point.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True)
    # print("model check point saved")
