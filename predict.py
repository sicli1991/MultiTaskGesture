import math
import tensorflow as tf
from tensorflow import keras
import pickle
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import os
from dataHandle import *
from myLayers import *
from keras.models import load_model

dataset_path = r"C:\Users\Sichao\Desktop\Jun25\v5\dataset_pickle"
input_data_channel = 1
# pickle_address = r"C:\Users\Sichao\Desktop\Jun25\v5\dataset_pickle\Test_data"
# test_x_address = os.path.join(r"dataset_pickle\Test_dataset", "test_X.pickle")
text_X = os.path.join(dataset_path, 'Test_data', 'test_X.pickle')
X_test = pickle_read_in(text_X)


def cal_dis(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_fingertip(tmp_test):
    i_max = 0.4
    cor_x = 0
    cor_y = 0
    for j in range(len(tmp_test)):
        for k in range(len(tmp_test[0])):
            if tmp_test[j][k] > i_max:
                i_max = tmp_test[j][k]
                cor_x = k
                cor_y = j
    # print("x: ", cor_x,"y: ", cor_y, "max: ", i_max)
    return (cor_x, cor_y)


def check_zero(p):
    if p[0] == 0 and p[1] == 0:
        return False
    else:
        return True


with open('modelGen/distillation_model_v3_1_Nov_07_17_47.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# wt_add = Wt_Add
loaded_model = model_from_json(loaded_model_json, custom_objects={'FM_subtract': fingermap_filter})
# model = load_model('modelGen/finalModel_v5_3.json', custom_objects={'Wt_Add': Wt_Add})

# load weights into new model
loaded_model.load_weights("modelGen/distillation_model_v3_1_Nov_07_17_47.h5")
print("Loaded model from disk")

print("test dataset loaded")

Y_pred = loaded_model.predict(X_test, verbose=1)
try:
    print(Y_pred.shape)
except:
    pass
# print(Y_pred.shape)
# pickle_out = open("predict_test_Y_YOLSE.pickle", "wb")
pickle_out = open(r"pickle_data\distillation_model_v3_1_Nov_07_17_47.pickle", "wb")
pickle.dump(Y_pred, pickle_out)
pickle_out.close()
print("predict finished!!!")

