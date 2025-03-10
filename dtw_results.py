from dtw import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import time


class DTWResults:
    def __init__(self, train_matrix_path):
        """
        This will be a class ONLY for the EKG data.

        Initializes the DTW Results with the training matrix and data path for 
        the original trainig.tsv file.

        Args:
            train_matrix: the training matrix, numpy array
             
        """
        # 1 first, load the training data but keep first row, call it train_data
        # 2 Then get classifications, 0 for R, 1 for A, etc. Use this for count.
        # 3 Then remove the first row
        # 4 Then load the training matrix
        
        
        # First, load the training data to extract count of each class
        self.train_matrix_path = train_matrix_path

        # First, get classifications before I remove the first row
        self.train_data = pd.read_csv(train_matrix_path, header=None)
        # Get the classifications from the training matrix
        # remove the '.' in label names, i.e. .1 and .2 and so on from row names
        # how do i access the first row of the df?
        #self.classifcations = self.train_data[0].unique()
        self.classifcations = self.train_data.iloc[0, :].unique()
        print(f"Classifications: {self.classifcations}")
        self.num_classes = len(self.classifcations)
        print(f"Number of classes: {self.num_classes}")
        # Count occurances of each class in the training data
        self.count = self.train_data.iloc[0, :].value_counts().values.tolist()
        print(f"Occurances of each class in the training data: {self.count}")


        # Load training matrix
        self.train_matrix = pd.read_csv(train_matrix_path, skiprows = 1, header=None)
        self.train_matrix = self.train_matrix.to_numpy()
        print(f"Loaded training matrix with shape: {self.train_matrix.shape}")
        print(f"Number of training signals: {self.train_matrix.shape[1]}")

        # Initialize elapsed time
        self.elapsed_time = 0
        



    def compute_distance_matrix(self, class_index, test_matrix):
        """
        Compute the distance matrix between training and test samples using DTW.
        
        Args:
            test_matrix: the test matrix, numpy array

        Returns:
            Dist_matrix: the distance matrix, numpy array (num_train_samples, num_test_samples)
        """

        # Load test matrix
        #print(f"Loaded test matrix with shape: {test_matrix.shape}")
        print(f"Number of test signals: {test_matrix.shape[1]}")

        m, d = self.train_matrix.shape[1], test_matrix.shape[1]
        # print the class label for which the accuracy is being calculated
        print(f"Computing distance matrix for {m} training signals and {d} test signals for {self.classifcations[class_index]}.")




        # Initialize distance matrix
        Dist_matrix = np.zeros((m, d))

        start_time = time.time()

        with tqdm(total=m*d, desc="Computing DTW distances") as pbar:
        # Compute DTW distance for each pair of training and test samples
            for i in range(m):
                for j in range(d):
                    Dist_matrix[i, j] = dtw(self.train_matrix[:, i], test_matrix[:, j]).distance
                    pbar.update(1)
        
        end_time = time.time()
        self.elapsed_time = (end_time - start_time)/60
        self.elapsed_time = round(self.elapsed_time, 2)
        print(f"Elapsed time for DTW distance computation: {self.elapsed_time:.2f} min")
        
        return Dist_matrix

    
    def find_accuracy(self, class_index, test_matrix_path):
        """
        Compute the accuracy of the classification for a specific class using DTW.

        Args:
            class_index (integer): the index of the class to find the
                accuracy for (0-based)
            test_matrix: the test matrix, numpy array
        Returns:
            matrix_min: the minimum distance matrix, numpy array(# of class, # of test signals)
            matrix_ind: the index of the minimum distance, numpy array (# of test signals)
            acc: the accuracy of the classification, float
        """
        # Load test matrix
        test_matrix = pd.read_csv(test_matrix_path, skiprows = 1, header=None)
        test_matrix = test_matrix.to_numpy()
        print(f"Loaded test matrix with shape: {test_matrix.shape}")
        
        m, d = self.train_matrix.shape[1], test_matrix.shape[1] # number of training and test signals
        dist_matrix = self.compute_distance_matrix(class_index, test_matrix)
        print(f"Shape of Dist_matrix: {dist_matrix.shape}, expected ({m}, {d})")

        # Split the distance matrix into submatrices for each class
        start = 0
        submatrices = []
        for size in self.count:  # for each class
            submatrices.append(dist_matrix[start:start+size, :]) # append the submatrix to the list
            start += size
        for sub in range(len(submatrices)):
            print(f"Submatrices sizes: {submatrices[sub].shape}")
        # Find the minimum value in each column of each submatrix
        matrix_min = []
        for sub in range(len(submatrices)):
            submatrices_min = np.reshape(np.min(submatrices[sub], axis=0), (1, d))
            print(f"submatrix shape: {submatrices_min.shape}, should be (1, {d})")
            matrix_min.append(submatrices_min)
        matrix_min = np.concatenate(matrix_min, axis=0)
        print(f"Shape of matrix_min: {matrix_min.shape}")
        # Find the index of the minimum value in each column
        matrix_ind = np.argmin(matrix_min, axis=0)
        # Calculate accuracy
        acc = np.sum(matrix_ind == class_index) / d
        acc = float(acc)
        acc = round(acc, 4)
        print(f"Accuracy of classification for class {class_index} is {acc:.4f}")

        print(f"Returning: matrix_min, matrix_ind, acc, self.elapsed_time, self.classifcations[class_index]") 
        print(f"Values: {matrix_min.shape}, {matrix_ind.shape}, {acc}, {self.elapsed_time}, {self.classifcations[class_index]}")

        return matrix_min, matrix_ind, acc, self.elapsed_time, self.classifcations[class_index]
    


###### Example usage: ##########
"""
path = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/2_current_filtered_data/all_train_data.csv'
classifier = DTWResults(path)
#path_test = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/R_test_matrix.csv'
#classifier.find_accuracy(0, path_test)
path_test = '/Users/vickyhaney/Documents/GAship/DrBruno/EKG/current_filtered_data/J_test_matrix.csv'
_, _, acc_J, J_time, Class = classifier.find_accuracy(6, path_test)
print(classifier.elapsed_time)
print(classifier.classifcations[6])
print(J_time)
print(Class)
print(acc_J)
"""

