import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from math import sqrt


def compute_accuracy(predicted_y, y):
    return np.sum(predicted_y == y)/y.shape[0]

if __name__ == '__main__':
    # Get data from file
    print("Start reading csv files")
    trainingData = pd.read_csv("CleanedData/train_reviews_combined_hl.csv")
    validData = pd.read_csv("CleanedData/validate_queries_combined_hl.csv")
    testData = pd.read_csv("CleanedData/test_queries_combined_hl.csv")
    print("Finish reading csv files")
    
    trainingData = trainingData.reset_index()
    validData = validData.reset_index()
    testData = testData.reset_index()
    #Y_column_index = trainingData.shape[1] - 1;
    
    # Remove unused attributes
    unused_attr_training = ["text","Unnamed: 0","index"];
    trainingData.drop(unused_attr_training,axis = 1, inplace = True);
    unused_attr_valid = ["Unnamed: 0_y","Unnamed: 0_x","index"];
    validData.drop(unused_attr_valid, axis = 1, inplace = True);
    unused_attr_test = ["Unnamed: 0","index"];
    testData.drop(unused_attr_test, axis = 1, inplace = True);
    # Split in to X_train, Y_train and X_valid , Y_valid
    # Y_column_index = Y_column_index - len(unused_attr);
    
    Y_train = trainingData[["review_stars"]];
    X_train = trainingData.drop({"review_stars"},axis = 1, inplace = False);
    X_train = np.nan_to_num(X_train);
        
    Y_valid = validData[["review_stars"]].values;
    X_valid = validData.drop({"review_stars"},axis = 1, inplace = False);
    X_valid = np.nan_to_num(X_valid);
    
    X_test = testData.iloc[: , :].values;
    X_test = np.nan_to_num(X_test);

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(X_valid.shape)
    print(Y_valid.shape)
    
    # Create classifier 
    
    # Linear Regression
    linearclf = LinearRegression();
    linearclf.fit(X_train,Y_train);
    Y_predicted = linearclf.predict(X_valid);
    train_error = sqrt(metrics.mean_squared_error(Y_valid, Y_predicted)) ;
    accuracy = compute_accuracy(np.rint(Y_predicted),Y_valid)
    Y_training_predict = linearclf.predict(X_train);
    train_error_train = sqrt(metrics.mean_squared_error(Y_train, Y_training_predict)) ;
    print("Train error for Linear Regression (MSE): %f"%train_error_train);
    print("Test error for Linear Regression (MSE): %f"%train_error);
    print("Accuracy is %f"%accuracy)
#
#    y_test = linearclf.predict(X_test);
#    
#    y_output = pd.DataFrame(y_test,columns = ["stars"])
#    y_output.index.name = "index"
#    y_output.to_csv("Submission.csv", index = "index")
    
    # KNN using KNN regression model
    
    knnclf = neighbors.KNeighborsRegressor(n_neighbors=10,weights = 'uniform');
    knnclf.fit(X_train,Y_train);
    Y_predicted = knnclf.predict(X_valid);
    train_error = sqrt(metrics.mean_squared_error(Y_valid, Y_predicted)) ;
    accuracy = compute_accuracy(np.rint(Y_predicted),Y_valid)
    Y_training_predict = knnclf.predict(X_train);
    train_error_train = sqrt(metrics.mean_squared_error(Y_train, Y_training_predict)) ;
    print("Train error for Linear Regression (MSE): %f"%train_error_train);
    print("Test error for Linear Regression (MSE): %f"%train_error);
    print("Accuracy is %f"%accuracy)
    y_test = knnclf.predict(X_test)
#    lowest_error = train_error
#    optimal_k = 1;
#    best_y_test = y_test
#    for k in range (2,20):
#        knnclf = neighbors.KNeighborsRegressor(n_neighbors=k,weights = 'distance');
#        knnclf.fit(X_train,Y_train);
#        knnclf.fit(X_train,Y_train);
#        Y_predicted = knnclf.predict(X_valid);
#        train_error = metrics.mean_squared_error(Y_valid, Y_predicted) ;
#        print("Train error for KNN Regression (MSE): %f"%train_error);
#        y_test = knnclf.predict(X_test)
#        if (lowest_error > train_error):
#            lowest_error = train_error
#            optimal_k = k;
#            best_y_test = y_test
#
#    
    y_output = pd.DataFrame(y_test,columns = ["stars"])
    y_output.index.name = "index"
    y_output.to_csv("Submission.csv", index = "index")