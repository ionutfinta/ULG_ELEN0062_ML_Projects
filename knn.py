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
#from sklearn.cross_validation import cross_val_score
# (Question 2)

# Put your funtions here
# ...


if __name__ == "__main__":
    # Put your code here
    
    #CREATION OF THE UNBALANCED DATASET
    
    X, y = make_unbalanced_dataset(3000) 
    
    #DIVIDING INTO LEARNING AND TESTING SET
    
    #Ici on regarde juste les tailles
    # Size of my data
    print(X.shape)
    print(y.shape)

    # Number of samples
    print(X.shape[0])
    # Number of attributes
    print(X.shape[1])
    
    
    number_of_ls = 1000 #Number of learning samples
    ## randomise a bit the data
    random_state = 0
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)
    a_permutation = random_state.permutation(np.arange(X.shape[0]))
    print("New order: {}".format(a_permutation))
    X = X[a_permutation, :]
    y = y[a_permutation]

    # Now I can divide the dataset
    X_train, X_test = X[:number_of_ls,:], X[number_of_ls:,:]
    y_train, y_test = y[:number_of_ls], y[number_of_ls:]

    print('LS size = {}'.format(X_train.shape))
    print('TS size = {}'.format(X_test.shape)) 
    
    #LEARN A MODEL = FIT THE MODEL PARAMETERS
    
    clf= KNeighborsClassifier(n_neighbors=1) #We instanciate a KNeighbors object
                                            #n_neighbors=1,5,50,100 and 500
    clf.fit(X_train, y_train)

    #DECISION BOUNDARY
   
    plot_boundary("TestBoundary(1n)", clf, X_test, y_test, 0.1, title="TestBoundary(1n)")
    
    y_pred=clf.predict(X_test)
    
    score1=accuracy_score(y_test, y_pred)
    
    print(score1)
    
    #Fin pour la Q1
    
    #10-FOLD-CROSS VALIDATION
  
    nb_neighbors=[1,5,50,100,500]
    mean_scores=[]
    for nb in nb_neighbors:
        
        knn=KNeighborsClassifier(n_neighbors=nb)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        mean_tmp=scores.mean() #mean of the scores of the 10 cv
        mean_scores.append(mean_tmp)  #on met ds ce vecteur les moyennes pour chaque nb
    
    
    max_mean_scores = max(mean_scores)
    print("Le score max vaut :")
    print(max_mean_scores)
    max_index = mean_scores.index(max_mean_scores)
    print("Ce qui correspond à l'indice :")
    print(max_index)
    
    #Fin pour la Q2
    
    #Fixed test set of size 500
    
    #matrice et vecteur tests fixes de 500 éléments
    X_test_q3=X[2500:,0:] #PROBLEME, ils mettent 500 éléments pourtant 2999-2500=499????
    y_test_q3=y[2500:]    
   
    N_LS=[50,150,250,350,450,500]
    
    for N in N_LS: #grande boucle sur la taille du LS
        
        for k in range(1,N+1): #boucle sur k
            knn_q3= KNeighborsClassifier(n_neighbors=k) #1 classifier pour chaque k
            
            for i in range (1,11) #boucle pour fit knn sur les 10 LS + leur création
            
            
            
            
        
    
    
    
    
    
    
    
    
    
    pass
