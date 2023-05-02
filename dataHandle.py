import pickle
import yaml
import os
from keras.utils.vis_utils import plot_model
from datetime import datetime
# from datetime import date


def pickle_read_in(path):
    with (open(path, "rb")) as openfile:
        while True:
            try:
                data = pickle.load(openfile)
            except EOFError:
                break
    return data


def config_read_in(path='config.yaml'):
    with open(path) as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config


def save_model_structure(model, now, ext='.png', override=True, from_train=True):
    # save model structure in figure
    # path = r'..\modelGen'
    config = config_read_in()
    model_name = config['model_name']
    de = now.strftime("%b_%d_%H_%M")  # %d/%m/%Y %H:%M:%S
    if from_train:
        file_name = model_name + '_' + de + ext
    else:
        file_name = model_name + ext
    file_path = os.path.join(config['model_save_path'], file_name)
    print(file_path)
    if not override:
        try:
            if os.path.exists(file_path):
                raise NameError
            else:
                pass
        except NameError:
            print(file_path, " already exists!!!!!")
            return False
    try:
        plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True)
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return False
    return True


def model_to_json(model, date, save_fig=True):
    config = config_read_in()
    model_path = config["model_save_path"]
    # now = datetime.now()
    de = date.strftime("%b_%d_%H_%M")  # %d/%m/%Y %H:%M:%S

    json_name, weight_name = config['model_name'] + '_' + de + '.json', config['model_name'] + '_' + de + '.h5'
    json_path = os.path.join(model_path, json_name)
    weight_path = os.path.join(model_path, weight_name)

    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weight_path)
    if save_fig:
        save_model_structure(model, date)
    else:
        pass

    print("model saved")
