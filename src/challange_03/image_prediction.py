import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, confusion_matrix, classification_report, roc_curve


path = "/home/prakash/Titan/padded_equalize"
directory_hash_dic = {}

def get_all_images(path):
    included_extenstions = ['jpg', 'bmp', 'png', 'jpeg']
    file_names = [fn for fn in os.listdir(path)
        if any(fn.endswith(ext) for ext in included_extenstions)]
    return  file_names

def read_img(path, name):
    return plt.imread(path + "/" + name)

def flatten_image(img):
    return img.flatten()  

def ravel_image(img):
    return img.ravel()  

def get_directories(path):
    return os.listdir(path) 

def get_train_test_data(features, lables):
    X_train, X_test, y_train, y_test = train_test_split(features, lables, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def get_trained_model(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #print("clf: ", clf) 
    return clf  

def predict(model, features_x):
    preds = model.predict(features_x)
    return preds

def get_accuracy(preds, features_y):
    accuracy = accuracy_score(features_y,preds)
    print("Accuracy:", accuracy)    
    return accuracy

def print_classification_report(y_true, y_preds, digits):
    print("printing classification_report... ")
    print(classification_report (
            y_true = y_true, 
            y_pred = y_preds,
            #target_names = list("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"),
            digits = digits
        )
    )

def print_confusion_matrix(y_true, y_pred):
    print("printing confusion_matrix... ")
    print(confusion_matrix(
        y_true = y_true, 
        y_pred = y_pred
    ))    


def main():
    features = []
    lables = []
    dataset = []

    directories = get_directories(path)
    for directory in directories:
        directory_hash = hash(directory)
        directory_hash_dic[directory] = directory_hash

        images = get_all_images(path + "/" + directory)
        for image in images:
            img = read_img(
                path = path + "/" + directory,
                name = image
            )
            raveled_image = ravel_image(img)
            features.append(raveled_image)
            lables.append(directory_hash)

    features = np.array(features)
    lables = np.array(lables)            
    X_train, X_test, y_train, y_test = get_train_test_data(
                        features = features,
                        lables = lables
                    )
    
    model = get_trained_model(
        X_train = X_train,
        y_train = y_train
    )

    preds = predict(
        model = model,
        features_x = X_test
    )

    get_accuracy(
        preds = preds,
        features_y = y_test
    )

    print_classification_report(
       y_true = y_test, 
       y_preds = preds, 
       digits = 2
       )

    print_confusion_matrix(
        y_true = y_test, 
        y_pred = preds
    )

    return model

if __name__ == "__main__":
    main()

