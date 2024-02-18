

def count_peaks(smoothed_series):
    peak_count = 0
    for i in range(5, len(smoothed_series) - 5):
        if smoothed_series[i] > max(smoothed_series[i-5:i]) and smoothed_series[i] > max(smoothed_series[i+1:i+6]):
            peak_count += 1
    return peak_count

def process_dataset(dataset):

    import numpy as np
    from scipy.stats import mode
    from scipy.ndimage.filters import gaussian_filter1d
    data_point_num = 0
    # Initialize the 2D dataset with zeros, now with 12*(6+50) features for each sample
    processed_dataset = np.zeros((dataset.shape[0], 12 * (6 + data_point_num)))  # 12 features, (6 stats + 50 time points) each
    
    for sample_idx in range(dataset.shape[0]):
        for feature_idx in range(dataset.shape[1]):
            # Extract the time series for the current sample and feature
            time_series = dataset[sample_idx, feature_idx, :]
            
            # Calculate statistical features
            min_val = np.min(time_series)
            max_val = np.max(time_series)
            std_dev = np.std(time_series)
            avg_val = np.mean(time_series)
            mode_val = mode(time_series).mode[0]
            
            # Smooth the time series
            smoothed_series = gaussian_filter1d(time_series, sigma=1)
            
            # Count peak points
            peak_count = count_peaks(smoothed_series)
            
            # Store the features in the processed dataset
            base_idx = feature_idx * (6 + data_point_num)
            processed_dataset[sample_idx, base_idx:base_idx + 6] = min_val, max_val, std_dev, avg_val, mode_val, peak_count
            
            # Add the first 50 time points
            processed_dataset[sample_idx, base_idx + 6:base_idx + data_point_num + 6] = time_series[:data_point_num]
            
    return processed_dataset




def one_hot_to_indices(one_hot_labels):
    """
    Convert a 2D numpy array of one-hot encoded labels to a 1D array of label indices.
    
    :param one_hot_labels: 2D numpy array where each row is a one-hot encoded label
    :return: 1D numpy array of label indices
    """
    import numpy as np
    # Use np.argmax to find the indices of the maximum values (1s in one-hot encoding) along axis 1
    label_indices = np.argmax(one_hot_labels, axis=1)
    label_indices = np.array([str(l) for l in label_indices])
    return label_indices





def convert_to_binary_labels(label_indices, positive_class_index):
    """
    Convert multiclass labels to binary labels. The class with index positive_class_index
    becomes 1, and all other classes become 0.
    
    :param label_indices: 1D numpy array of multiclass label indices
    :param positive_class_index: The index of the class to be considered as '1' (positive)
    :return: 1D numpy array of binary labels
    """
    import numpy as np
    # Initialize binary labels array with zeros
    binary_labels = np.zeros(label_indices.shape, dtype=int)
    
    # Set elements to 1 where the label_indices match the positive_class_index
    binary_labels[label_indices == positive_class_index] = 1
    
    binary_labels = np.array([str(l) for l in binary_labels])
    return binary_labels


def count_label_occurrences(label_indices):
    """
    Count the number of occurrences of each label in the label_indices array.
    
    :param label_indices: 1D numpy array of label indices
    :return: 1D numpy array where the ith element is the number of occurrences of label i
    """
    import numpy as np
    # Use np.bincount to count occurrences of each label
    #label_indices_n = np.array([int(l) for l in label_indices])
    #label_counts = np.bincount(label_indices_n)

    # Get unique labels and initialize a count dictionary
    unique_labels = np.unique(label_indices)
    count_dict = {label: 0 for label in unique_labels}
    
    # Count occurrences of each label
    for label in label_indices:
        count_dict[label] += 1
    
    # Convert the count dictionary to a list of counts
    label_counts = [count_dict[label] for label in unique_labels]
    
    return np.array(label_counts)


def load_12ECG_model_sk(input_directory,model_name):
    import joblib, os
    # load the model from disk 
    f_out=f'{model_name}_model.sav'
    filename = os.path.join(input_directory,f_out)

    loaded_model = joblib.load(filename)
    loaded_model['classes'] = [str(l) for l in loaded_model['classes']]
    return loaded_model



def create_confusion(y_true,y_pred):


    import numpy as np

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # convert '1' --> '01'
    y_true_t = y_true



    classes = np.unique(y_true_t)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add labels to each cell
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.savefig('confusion.png')



def train_sklearn(headers_datasets, output_directory, fDatas,model_name,train_en,binary_class,raw_en):
    import sys
    from manipulations import get_scored_class, get_name, cv_split
    from global_vars import labels, equivalent_mapping, Dx_map, Dx_map_unscored

    from global_vars import normal_class, weights, disable_tqdm, enable_writer, run_name, n_segments, max_segment_len, epoches
    from resnet1d import ECGBagResNet
    from dataset import BagSigDataset
    from myeval import agg_y_preds_bags, binary_acc, geometry_loss, compute_score
    from imbalanced_weights import inverse_weight
    from pytorchtools import EarlyStopping, add_pr_curve_tensorboard
    from saved_data_io import read_file 

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import torch
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    if enable_writer:
        from torch.utils.tensorboard import SummaryWriter
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler

    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    import os
    from sktime.classification.kernel_based import RocketClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF

    Codes, dataset_train_idx, dataset_test_idx, filenames = cv_split(headers_datasets)

    print("len Codes:",len(Codes))
    print(" Codes:",Codes[:3])
    print("dataset_test_idx:",dataset_test_idx)
    print(labels)

    datasets = np.sort(list(headers_datasets.keys()))

    #global labels
    # agg labels
    data_img2_labels = []
    for i in tqdm(range(len(Codes)), disable=disable_tqdm):
        data_img2_labels.append(get_scored_class(Codes[i], labels))
    #print("data_img2_labels",data_img2_labels)
    data_img2_labels = np.array(data_img2_labels)
    assert len(data_img2_labels) == len(Codes)

    # change to equivalent mapping
    for key in equivalent_mapping.keys():
        print('equivalent', key, equivalent_mapping[key])
        #print(int(key))
        #print(np.argwhere(labels==int(key)))
        #print(np.argwhere(labels==int(key)).flatten())
        #if np.argwhere(labels==int(key)) != []:
        try:
            
            key_idx = np.argwhere(labels==int(key)).flatten()[0]
            val_idx = np.argwhere(labels==int(equivalent_mapping[key])).flatten()[0]
            key_pos = np.argwhere(data_img2_labels[:,key_idx]==1).flatten()
            val_pos = np.argwhere(data_img2_labels[:,val_idx]==1).flatten()
            data_img2_labels[key_pos,val_idx] = 1
            data_img2_labels[val_pos,key_idx] = 1
            print(int(key))
        except:
            pass
    del Codes, dataset_train_idx, dataset_test_idx, headers_datasets

    names = [get_name(label, Dx_map, Dx_map_unscored) for label in labels]
    class_idx = np.argwhere(np.sum(np.array(data_img2_labels),axis=0)!=0).flatten() 
    names = np.array(names)[class_idx]
    print("names:",names)
    print("class_idx",class_idx)
    print("labels:",labels)
    normal_idx = np.argwhere(labels[class_idx]==int(normal_class)).flatten()[0]

    print("#classes: ", len(class_idx), "data_img2_labels.dim", data_img2_labels.shape)
    print(data_img2_labels[:2])
    print("normal_idx: ", normal_idx)

    # get device
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(0)

    st = time.time()

    train_class_weight = torch.Tensor(inverse_weight(data_img2_labels, class_idx)).to(device)
    print("train_class_weight", train_class_weight)

    sig_datasets_train = BagSigDataset(fDatas, data_img2_labels, class_idx, 'train', n_segments, max_segment_len)

    trainDataset = torch.utils.data.Subset(sig_datasets_train, list(range(len(sig_datasets_train))))

    batch_size = 64
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=0)





    # Initialize lists to store features and labels
    features_list = []
    labels_list = []

    # Assume trainLoader is your DataLoader instance
    for batch_idx, (features, labels) in enumerate(trainLoader):
        # Move your features and labels to CPU and convert them to NumPy arrays if they are in torch.Tensor format
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Append the features and labels to the respective lists
        features_list.append(features)
        labels_list.append(labels)

    # Concatenate all features and labels
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    print("X:")
    print(X.shape)
    #print(len(X))
    #print(len(X[0]))
    #print(X[:2])
    # Replace NaN values with mean values
    #imputer=SimpleImputer().fit(X)
    #X=imputer.transform(X)

    # Assuming your data is in a 4D array named 'data_4d'
    # Shape of data_4d: (num_samples, dim1, dim2, dim3)
    num_samples = X.shape[0]
    print("num_samples:",num_samples)
    # Reshape data into 2D
    X = np.squeeze(X, axis=1)
    if model_name!='roc':
        if raw_en=='raw':
            X = X.reshape(num_samples, -1) # raw 2d
        else:
            X = process_dataset(X) # 2d with features of features
    #elif model_name=='roc':
    #    X = np.squeeze(X, axis=1)
    
    y = one_hot_to_indices(y)


    print("X reshape:")
    print(X.shape)
    #print(len(X))
    #print(len(X[0]))
    print(X[:2])


    print('y')
    print(y.shape)
    print(y[:10])

   

    if binary_class=='bin':
        y = convert_to_binary_labels(y,str(normal_idx))

    


    test_size= 2600
    X_train = X[:test_size]
    y_train = y[:test_size]

    X_test = X[test_size:]
    y_test = y[test_size:]

    print("y_train:", y_train[:10])
    print("y_test:", y_test[:10])

    label_counts = count_label_occurrences(y_train)
    print("y_train distribution:", label_counts) 
    label_counts_t = count_label_occurrences(y_test)
    print("y_test distribution:", label_counts_t) 




    if model_name=='rf':
        # Initialize the RandomForestClassifier
        model = RandomForestClassifier(n_estimators=500, random_state=0)
    elif model_name=='roc':
        model = RocketClassifier(num_kernels=1000)
    elif model_name=='knn':
        model = KNeighborsClassifier(n_neighbors=6)
    elif model_name=='mlp':
        if raw_en=='raw':
            model = MLPClassifier(hidden_layer_sizes=(2500,500,100,20,))
        else:
            #model = MLPClassifier(hidden_layer_sizes=(150,70,20))
            #model = MLPClassifier(hidden_layer_sizes=(72,35,))
            model = MLPClassifier(hidden_layer_sizes=(400,100,20,))
    elif model_name=='svm':
        #model = LinearSVC(max_iter=2000)
        model = LinearSVC()

    '''
    elif model_name=='svc':
        model = SVC(gamma=2, C=1, random_state=42)
    elif model_name=='gus':
        kernel = 1.0 * RBF(1.0)
        model = GaussianProcessClassifier(kernel=kernel,random_state=0)
    '''



    import datetime
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    print('time_str',time_str)


    if train_en == 'train':
        print("fitting model ...")
        # Fit the model
        model = model.fit(X_train, y_train)

        # Save model.
        print('Saving model...')

        final_model={'model':model,'classes':labels}

        filename = os.path.join(output_directory, f'{model_name}_model.sav')
        joblib.dump(final_model, filename, protocol=0)
    
    else:
        loaded_model = load_12ECG_model_sk(output_directory,model_name)
        model = loaded_model['model']
        classes = loaded_model['classes']

    if train_en == 'train':
        print('Evaluation ..')
        # Make predictions on the training data
        train_predictions = model.predict(X_train)
        print("train_predictions")
        print(train_predictions.shape)
        print(train_predictions[:10])
        from sklearn.metrics import accuracy_score
        # Calculate accuracy on the training data
        train_accuracy = accuracy_score(y_train, train_predictions)
        print(f'Accuracy on training data: {train_accuracy}')


    import time



    start_time = time.time()
    # Make predictions on the test data
    train_predictions = model.predict(X_test)
    end_time = time.time()

    print(f"Processed batch {X_test.shape[0]}, Time: {end_time - start_time:.4f} seconds")

    print("test_predictions")
    print(train_predictions.shape)
    print(train_predictions[:10])
    from sklearn.metrics import accuracy_score
    # Calculate accuracy on the training data
    train_accuracy = accuracy_score(y_test, train_predictions)
    print(f'Accuracy on test data: {train_accuracy}')

    label_counts_p = count_label_occurrences(train_predictions)
    print("y_prediction distribution:", label_counts_p) 

    if binary_class=='bin':
        count = 0
        for i in range(len(y_test)):
            if (y_test[i]=='1') and (train_predictions[i]=='1'):
                count += 1
        
        avg = count/label_counts_t[1]
        print("real normal acc is : ", avg)

    create_confusion(y_test,train_predictions)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    print('time_str',time_str)