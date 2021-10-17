"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score


#function used in question 3
def creation_LS(X,y,N):
    """Generates a random learning set of size N from the data in X 
    (containing the input samples) and in y (containing the corresponding 
    output values).                                         

    Parameters
    ----------
    X: array containing the input samples
    y: array containing the corresponding output values   
    
    Return
    ------
    X_random_rows : array of shape [N, (number of columns of X)]
    y_random_rows : array of shape [N]
    
    """
    number_of_rows = X.shape[0]
    random_indices = np.random.choice(number_of_rows, size=N, replace=False)
    X_random_rows = X[random_indices, :]
    y_random_rows=  y[random_indices]
    return X_random_rows, y_random_rows


if __name__ == "__main__":
    
    #Creation of the unbalanced datatset
    
    X, y = make_unbalanced_dataset(3000) 
    
    #Dividing into learning and testing set
       
    number_of_ls = 1000    #Number of learning samples
    #Randomization of the data
    random_state = 0
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)
    a_permutation = random_state.permutation(np.arange(X.shape[0]))
    X = X[a_permutation, :]
    y = y[a_permutation]

    #Dividing of the dataset
    X_train, X_test = X[:number_of_ls,:], X[number_of_ls:,:]
    y_train, y_test = y[:number_of_ls], y[number_of_ls:]
    
    ################ QUESTION 1 ################
    
    N_neighbors=[1,5,50,100,500]
    
    for N_neigh in N_neighbors:
        clf= KNeighborsClassifier(n_neighbors= N_neigh) # We instanciate a KNeighbors object                                                                                       
        clf.fit(X_train, y_train) #We fit the model 

    #Decision boundary
        name= 'Boundary %d' %N_neigh        
        title= 'Decision boudary for k=%d' %N_neigh        
        plot_boundary(name, clf, X_test, y_test, 0.1, title=title)
    
    ################ QUESTION 2 ################
  
    nb_neighbors=[1,5,50,100,500]
    mean_scores=[]
    for nb in nb_neighbors:
        knn=KNeighborsClassifier(n_neighbors=nb)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        mean_tmp=scores.mean() #mean of the scores of the 10 cv
        mean_scores.append(mean_tmp)  #vector containing the mean scores for each nb
        
    max_mean_scores = max(mean_scores)
    print("The mean accuracy of the optimal value of n_neighbors is :")
    print(max_mean_scores)
    max_index = mean_scores.index(max_mean_scores)
    print("The optimal value of n_neighbors is %d : " %nb_neighbors[max_index])
    
    ################ QUESTION 3 ################       
    
    #Test set of size 500
    X_test_q3=X[2500:,0:] 
    y_test_q3=y[2500:]    
    #All other samples that will be used to produce the LS
    X_train_q3=X[:2500,0:]
    y_train_q3=y[:2500]
    
    N_LS=[50,150,250,350,450,500] #sizes of the LS
    optimal_value=[] #array that will contain the optimal k for each N
        
    
    #For each N I have a vector score_N that I want to draw
    
    
    for N in N_LS: #loop on the size of the LS

        score_N=[] #array that will contain the scores for each k

        for k in range(1,N+1):
            score_k=0                                
            for i in range (1,11): #loop for the 10 different LS
                knn_q3= KNeighborsClassifier(n_neighbors=k)
                X_q3_LS, y_q3_LS=creation_LS(X_train_q3,y_train_q3,N)
                knn_q3.fit( X_q3_LS, y_q3_LS)                

                y_pred_q3=knn_q3.predict(X_test_q3)   
                score_tmp=accuracy_score(y_test_q3, y_pred_q3)                
                score_k= score_k + score_tmp #score_k will be the total of the 10 scores
                
            
            score_k= score_k/10
            score_N.append(score_k)
                            
        optimal_value_N=max(score_N)
        optimal_value_N_index=score_N.index(optimal_value_N)
        optimal_value.append(optimal_value_N_index)
        if N==50:
            x_plot = range(1,51)
            plt.plot(x_plot,score_N, color='b')
        elif N==150:
            x_plot = range(1,151)
            plt.plot(x_plot,score_N, color='g')    
        elif N==250:
            x_plot = range(1,251)
            plt.plot(x_plot,score_N, color='r')    
        elif N==350:
            x_plot = range(1,351)
            plt.plot(x_plot,score_N, color='c')    
        elif N==450:
            x_plot = range(1,451)
            plt.plot(x_plot,score_N, color='m')    
        elif N==500:
            x_plot = range(1,501)
            plt.plot(x_plot,score_N, color='y')
            plt.xlabel('number of neighbors')
            plt.ylabel('average accuracy')
            plt.suptitle('Evolution of mean test accuracies', fontsize=11)
            
    #k optimal en fonction de N          
    plt.figure()
    plt.plot(N_LS, optimal_value, 'o', color='black')
    plt.xlabel('training set size') 
    plt.ylabel('number of neighbors')
    plt.suptitle('Evolution of the optimal value of the number of neighbors', fontsize=11)
                            
    pass
