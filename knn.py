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


def creation_LS(X,y,N):
    number_of_rows = X.shape[0]
    random_indices = np.random.choice(number_of_rows, size=N, replace=False)
    X_random_rows = X[random_indices, :]
    y_random_rows=  y[random_indices]
    return X_random_rows, y_random_rows


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
    
    print("QUESTION 1 QUESTION 1 QUESTION 1 ")
    #LEARN A MODEL = FIT THE MODEL PARAMETERS
    
    N_neighbors=[1,5,50,100,500]
    
    for N_neigh in N_neighbors:
        clf= KNeighborsClassifier(n_neighbors= N_neigh) 
                                            #We instanciate a KNeighbors object
                                            #n_neighbors=1,5,50,100 and 500
        clf.fit(X_train, y_train)

    #DECISION BOUNDARY
        name= 'Boundary %d' %N_neigh
        #print(name)
        title= 'Decision boudary for k=%d' %N_neigh
        #print(title)
        plot_boundary(name, clf, X_test, y_test, 0.1, title=title)
    
    
    #score1=accuracy_score(y_test, y_pred)
    
    #print(score1)
    
    #Fin pour la Q1
    print("QUESTION 2 QUESTION 2 QUESTION 2 ")
    #10-FOLD-CROSS VALIDATION
  
    nb_neighbors=[1,5,50,100,500]
    mean_scores=[]
    for nb in nb_neighbors:
        
        knn=KNeighborsClassifier(n_neighbors=nb)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        mean_tmp=scores.mean() #mean of the scores of the 10 cv
        mean_scores.append(mean_tmp)  #on met ds ce vecteur les moyennes pour chaque nb
    
    
    max_mean_scores = max(mean_scores)
    print("The mean accuracy of the optimal value of n_neighbors is :")
    print(max_mean_scores)
    max_index = mean_scores.index(max_mean_scores)
    print("The optimal value of n_neighbors is %d : " %nb_neighbors[max_index])
    
    
    #Fin pour la Q2
    print("QUESTION 3 QUESTION 3 QUESTION 3 ")
    #Fixed test set of size 500
    
    #matrice et vecteur tests fixes de 500 éléments
    X_test_q3=X[2500:,0:] #PROBLEME, ils mettent 500 éléments pourtant 2999-2500=499????
    y_test_q3=y[2500:]    
   
    X_train_q3=X[:2500,0:]
    y_train_q3=y[:2500]
    
    N_LS=[50,150,250,350,450,500]
    optimal_value=[]
        
    
    #For each N I have a vector score_N that I want to draw
    
    
    for N in N_LS: #grande boucle sur la taille du LS

        score_N=[] #score_N est vide au début, on va y mettre les scores pour un N (pour chaque k)

        for k in range(1,N+1):#boucle sur k,ça va de 1 à N compris
            score_k=0 #score_k est le score pour un k
        
           
            
            for i in range (1,11):#boucle(de 1 à 10 compris)pour fit knn sur les 10 LS + leur création
                knn_q3= KNeighborsClassifier(n_neighbors=k) #1 classifier pour chaque k
                X_q3_LS, y_q3_LS=creation_LS(X_train_q3,y_train_q3,N)
                knn_q3.fit( X_q3_LS, y_q3_LS)
                
                y_pred_q3=knn_q3.predict(X_test_q3)   
                score_tmp=accuracy_score(y_test_q3, y_pred_q3)
                
                score_k= score_k + score_tmp #On additionne pour avoir le total des 10 scores
                
            
            score_k= score_k/10
            score_N.append(score_k)
            
        
        #PLOT #draw here, ici on a le score_N du N courant
        optimal_value_N=max(score_N)
        optimal_value.append(optimal_value_N)
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
            
    #Q3B, k optimal en fonction de N  plot dots          
    plt.figure()
    plt.plot(N_LS, optimal_value, 'o', color='black')
    plt.xlabel('training set size') 
    plt.ylabel('number of neighbors')
    plt.suptitle('Evolution of the optimal value of the number of neighbors', fontsize=11)
            
    
    
    
    
    pass
