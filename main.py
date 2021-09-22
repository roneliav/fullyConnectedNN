import pandas as pd
import numpy as np
from math import sqrt
import time
import os
import glob
import os.path

DATA_SIZE = 8000

def print_weights(weights):
    for w in range(len(weights)):
        print(f"from layer {w} to layer {w+1} weights:")
        print(weights[w])

def get_random_weights(laer_size_dict): #He-Normalize
    rows = laer_size_dict['rows']
    columns = laer_size_dict['columns']
    return sqrt(2 / (rows + columns)) * np.random.randn(rows, columns)

def get_output_layer_from_layers(layers):
    return layers[len(layers)-1]

def get_train_and_target():
    train_data = pd.read_csv("data\\train.csv", header=None)  # header=None means that there are not columns' names in the csv
    # divide to train and target data frames
    target = train_data.loc[:, 0]  # first column
    train_data = train_data.drop(columns=0)  # drop the first column
    train_data = train_data.rename(columns=lambda c: c - 1).to_numpy()
    return train_data, target

def get_predicted_result_from_output_layer(output_layer):
    return np.where(output_layer == output_layer.max())[0][0] + 1 # the targets begin from 1

def get_error_output(output_layer, target_of_this_row):
    target_layer = np.zeros(10)
    target_layer[target_of_this_row - 1] = 1
    error_output = target_layer - output_layer
    return error_output

def write_weights_to_csv(between_layers_weights, test_folder, epoch_number):
    os.mkdir(f"{test_folder}\\epoch_{epoch_number}")
    for weights in range(len(between_layers_weights)):
        pd.DataFrame(data=between_layers_weights[weights]).to_csv(f"{test_folder}\\epoch_{epoch_number}\\layer_{weights}_to_layer_{weights+1}_weights.csv")

def print_output_str(test_folder, epoch_number, correct_predict): # print accuracy
    output_str = f"in epoch_{epoch_number} the accuracy precents are {(correct_predict / 8000) * 100}%\n"
    print(output_str)
    with open(f"{test_folder}\\output_layer.txt", "a") as output_file:
        output_file.write(output_str)

def get_last_epoch_number():
    folder_count = 0
    for folders in os.listdir(test_folder):
        folder_count += 1  # increment counter
    try:
        v = open(f"{test_folder}\\validate.txt", 'r')  # if 'validate' file exist, decrease one
        v.close()
        folder_count -= 1
    except:
        pass
    try:
        v = open(f"{test_folder}\\output_layer.txt", 'r')  # if 'output' file exist, decrease one
        v.close()
        folder_count -= 1
    except:
        pass
    return folder_count - 2

def get_weights_from_epoch(epoch_number, num_of_weights):
    epoch_folder = 'epoch_' + str(epoch_number)
    between_layers_weights = {}
    for i in range(num_of_weights):
        between_layers_weights[i] = pd.read_csv(f"{test_folder}\\{epoch_folder}\\layer_{i}_to_layer_{i + 1}_weights.csv", index_col='Unnamed: 0').to_numpy()
    return between_layers_weights

def forward_propagation_one_level(layer, weights, last=False):
    next_layer = weights * layer[:, np.newaxis]
    next_layer = next_layer.sum(axis=0)
    next_layer = np.maximum(next_layer, 0)
    if not last:
        next_layer[next_layer.shape[0] - 1] = -1  # bias unit
    return next_layer

def backpropagation_one_weights(old_weights, layer, above_layer_error, lr):
    above_layer_rows = above_layer_error.shape[0]
    layer_n = layer
    layer_rows = layer_n.shape[0]
    # old_weights = old_weights.to_numpy()
    new_weights = old_weights + (layer_n.reshape(layer_rows, 1) * above_layer_error.reshape(1, above_layer_rows) * lr)
    # new_weights = pd.DataFrame(data=new_weights)
    error_layer = ((layer > 0) *1).reshape(layer_rows, 1) * ((old_weights * above_layer_error.reshape(1, above_layer_rows)).sum(axis=1)).reshape(layer_rows, 1)
    return new_weights, error_layer

def make_noise(layer, percent_of_noise=0.1):
    max_int = layer.shape[0]
    num_noise = int(percent_of_noise*max_int)
    list = np.random.randint(0, max_int, size=num_noise)
    np.put(layer, list, 0)
    return layer

def get_noise_for_this_row(epoch_number, conf_noise = {}):
    """
    should consider according conf_noise and the epoch number
    meanwhile, 2/3 lines will be with noise of 0.2 percent
    :return: percent of input neurons to be noised
    """
    if conf_noise:
        return conf_noise['percent']
    return 0

def full_forward_propagation(first_layer, between_layers_weights, noise_percent=0):
    num_of_weights = len(between_layers_weights)
    layers = {}
    layer_number = 0
    layers[layer_number] = first_layer  # one raw in csv
    if noise_percent:
        layers[layer_number] = make_noise(layers[layer_number], noise_percent)
    layers[layer_number] = np.append(layers[layer_number], [-1])  # bias unit
    for layer_number in range(1, num_of_weights + 1):
        layers[layer_number] = forward_propagation_one_level(layers[layer_number - 1], between_layers_weights[layer_number - 1], last=layer_number == num_of_weights)

    return layers

def full_back_propagation(between_layers_weights, error_output, layers, lr):
    num_of_weights = len(between_layers_weights)
    updated_weights = {}
    above_layer_error = error_output
    for layer_number in range(num_of_weights - 1, -1, -1):
        updated_weights[layer_number], above_layer_error = backpropagation_one_weights(
            between_layers_weights[layer_number], layers[layer_number], above_layer_error, lr)
    return updated_weights

def make_train(test_folder, lr, between_layers_weights, start_train_epoch, conf_noise={}):
    epoch_number = start_train_epoch
    train_data, target = get_train_and_target()

    """
    start training
    """
    while (True):  # one epoch for each loop
        correct_predict = 0
        start_time = time.time()
        for i in range(DATA_SIZE):
            row_number = i
            target_of_this_row = target[row_number]
            # forward propagation
            noise = get_noise_for_this_row(epoch_number, conf_noise)
            layers = full_forward_propagation(train_data[row_number], between_layers_weights, noise_percent=noise)
            output_layer = get_output_layer_from_layers(layers)
            predicted_result = get_predicted_result_from_output_layer(output_layer)

            """
            print and check output
            """
            print(f"excepted: {target_of_this_row}, got {predicted_result}")
            if target_of_this_row == predicted_result:
                correct_predict += 1

            # calculate output error
            error_output = get_error_output(output_layer, target_of_this_row)
            # backward propagation
            between_layers_weights = full_back_propagation(between_layers_weights, error_output, layers, lr)
            print(f"row {i},  {time.time() - start_time} second\n")

        # after full epoch - write accuracy precents and write weights to csvs
        print_output_str(test_folder, epoch_number, correct_predict)
        write_weights_to_csv(between_layers_weights, test_folder, epoch_number)
        epoch_number = epoch_number + 1

def full_train(test_folder, lr, weights_dict, conf_noise={}):
    num_of_weights = len(weights_dict)
    # initiate random weights and write them to csv
    between_layers_weights = {}
    epoch_number = -1 # for initiate weights
    for weights in range(num_of_weights):
        between_layers_weights[weights] = get_random_weights(weights_dict[weights])
    write_weights_to_csv(between_layers_weights, test_folder, -1)
    # make train from start
    make_train(test_folder, lr, between_layers_weights, start_train_epoch=0, conf_noise=conf_noise)

def get_validate_and_target():
    validate_data = pd.read_csv("data\\validate.csv", header=None)
    # divide to train and target data frames
    target = validate_data.loc[:, 0]  # first column
    print(target.shape)
    validate_data = validate_data.drop(columns=0)  # drop the first column
    validate_data = validate_data.rename(columns=lambda c: c - 1).to_numpy()
    print(validate_data)
    return validate_data, target

def get_last_validated_epoch():
    try:
        validate_file = open(f"{test_folder}\\validate.txt", "r")
        last_validated_epoch = int(validate_file.readlines()[-1].split()[1][6:])
        validate_file.close()
    except: # first validation
        last_validated_epoch = -1
    return last_validated_epoch # because start from epoch 0

def get_num_of_wights_from_test_folder(test_folder, epoch_number):
    csv_files = glob.glob(os.path.join(f"{test_folder}\\epoch_{epoch_number}", '*.csv'))
    return len(csv_files)

def validate(test_folder, epoch_number=None):
    if epoch_number==None:
        epoch_number = get_last_validated_epoch()
    num_of_weights = get_num_of_wights_from_test_folder(test_folder, epoch_number=epoch_number)
    validate_data, target = get_validate_and_target()
    rows_to_predict = validate_data.shape[0]
    output_file = open(f"{test_folder}\\validate.txt", "a")
    epoch_number += 1

    while(True):
        epoch_folder = 'epoch_' + str(epoch_number)
        correct_predict = 0
        """
        initiate weights from csv
        """
        if get_num_of_wights_from_test_folder(test_folder, epoch_number=epoch_number) != num_of_weights:
            break;
        between_layers_weights = get_weights_from_epoch(epoch_number, num_of_weights)

        """
        forward propagation for the whole epoch
        """
        for row in range(rows_to_predict): # for each row in the validation csv
            layers = full_forward_propagation(validate_data[row], between_layers_weights)
            predicted_result = get_predicted_result_from_output_layer(get_output_layer_from_layers(layers))
            print(f"row {row}:  excepted: {target[row]}, got {predicted_result}")
            if target[row] == predicted_result:
                correct_predict += 1

        output_str = f"in {epoch_folder} the accuracy precents are {(correct_predict/rows_to_predict)*100}%\n"
        print(output_str)
        output_file.write(output_str)

        epoch_number += 1
    output_file.close()

def train_from_middletest_folder(test_folder, lr, epoch_number=None):
    if epoch_number == None:
        csv_files = glob.glob(os.path.join(f"{test_folder}\\epoch_-1", '*.csv'))
        num_of_weights = len(csv_files)
        last_epoch = get_last_epoch_number()
    else:
        csv_files = glob.glob(os.path.join(f"{test_folder}\\epoch_{epoch_number}", '*.csv'))
        num_of_weights = len(csv_files)
        last_epoch = epoch_number
    # initiate weights from specific epoch
    between_layers_weights = get_weights_from_epoch(last_epoch, num_of_weights=num_of_weights)
    # print_weights(between_layers_weights)
    make_train(test_folder, lr, between_layers_weights, start_train_epoch=last_epoch+1)

# hyper-parameters
weights_dict = {0:{"rows": 3073, "columns": 2501}, # first hidden layer: 3073 input, 2501 output (including bias neuron)
                1:{"rows": 2501, "columns": 1537},
                2:{"rows": 1537, "columns": 10}}
lr = 0.001
test_folder = "test_name" # recomend that the name will tell something about the architecture.
os.mkdir(test_folder)
conf_noise = {'percent': 0.1}

# train
full_train(test_folder, lr, weights_dict, conf_noise)
# train_from_middletest_folder(test_folder, lr=0.00001, epoch_number=20)
# validate(test_folder)
