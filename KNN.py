import pandas as pd
import numpy as np 



#create pandas dataframe
#skip row 1 and make sure header is not read 
train_data = pd.read_csv('MNIST_training.csv', header=None, skiprows=1 )
test_data = pd.read_csv('MNIST_test.csv', header=None, skiprows=1)

#function to calculate the euclidean distance form
#change to float, since it was reading as incorrect type 
def euclideanDistance(a, b):
    distance = np.sqrt(np.sum((a.astype(int) - b.astype(int))**2))
    return distance

#training data : y : class labels ; x:properties of samples
y= np.array(train_data.iloc[:, 0] ) 
x =np.array (train_data.iloc[:, 0:-1])

#test data 
groundTruth_y = (test_data.iloc[:, 0]) 
test_x = np.array(test_data.iloc[:, 0:-1]) 

#KNN algorithm 
def KNN(K, testX, testY, trainX, trainY):
        #list to store predictions
        predictions = []
        correct = 0
        #loop through all test data samples 
        for i, test_sample in enumerate(testX):
                    #find distance between test samples and all training samples 
                    distances = np.array([euclideanDistance(test_sample, train_samples) for train_samples in trainX])
                   #sort the array indices, use K to find nearest neighbors
                    K_nearest_neighbors=distances.argsort()[:K]
                    #get the labels from the indices 
                    K_nearest_labels = [trainY[j] for j in K_nearest_neighbors]
                    #get the majority class by finding the highest frequency
                    majority = max(K_nearest_labels, key = K_nearest_labels.count)
                    #add majority to predictions list
                    predictions.append(majority)
        #loop through predictions made and compare with ground truth to find # of correct predictions
        for l in range(len(predictions)):
            prediction = predictions[l]
            correct_label = testY[l]
            if prediction == correct_label:
                correct +=1
        #find acurracy from # of correct predictions/total # test data
        accuracy= correct/len(testX)
        return accuracy

#test different k values 
K = [3,5,7,9]
for k in K:
    print (f"For K = {k}")
    accuracy =(KNN(k, test_x, groundTruth_y, x, y)) 
    print (f"Accuracy = {accuracy}")
    
    


             
            
                





           

