
import numpy as np




   
hidden_num = 50 # number of neurons in hidden layer
sigma = 10 # radial 
centers = []
weights = []

def kernel_function( center, data_point):
    return np.exp(-np.linalg.norm(center-data_point)**2/(sigma**2))#capital phi in pdf slide 11

def compute_matrix(X):#pdf slide 12
    global centers,data_point
    matrix = np.zeros((len(X), hidden_num))#tedad e voroodr * tedad phi ha :)
    for i in range(len(X)):
        for j in range(len(centers)):
            matrix[i][j] = kernel_function(centers[j], X[i])#bar asa e har phi va har voroodi ye deraye khahim dasht
    return matrix

def select_centers(X):#randomly center choose between data_list points 
    temp=[]
    for i in range(hidden_num):
        temp.append(np.random.choice(len(X)))    
    return temp

def fit(X, Y):#make network ready
    global centers
    global G
    global weights

    centers = select_centers(X)
    G = compute_matrix(X)
    #weights = [1 for i in range(hidden_num)]
    weights = np.dot(np.linalg.pinv(G), Y)#pinv returns the inverse of your matrix
    # ***** it is refer to pdf slide 13

def predict(X):#pdf slide 5
    #bar asas e voroodimoon ye matris misazim bed pishbini miknim
    global weights
    matrix = compute_matrix(X)
    predictions = np.dot(matrix, weights)
    return predictions
    


#*******************************************************
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-2, 2, 1000)
y = x*x*x
fit(x, y)
x = np.linspace(-2, 2, 1000)
y_pred = predict(x)

plt.plot(x, y, 'b-', label='real')
plt.plot(x, y_pred, 'r-', label='fit')
plt.legend(loc='upper right')
plt.title('Interpolation using a RBFN')
plt.show()


