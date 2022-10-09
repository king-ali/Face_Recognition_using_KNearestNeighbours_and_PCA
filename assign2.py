import numpy as np
from sklearn.model_selection import train_test_split
import random as randd
from scipy.spatial import distance as da
import time
from matplotlib import pyplot as plt
import panda as pd

fea = pd.read_csv('fea.csv')

fea = fea/np.linalg.norm(fea, axis = 1, keepdims=True)

def Split(fea,dimensions,train_imgn,test_imgn,Total,Classes):
 training_data = np.empty([train_imgn * Classes, dimensions], dtype=float)
 training_label = np.empty([train_imgn * Classes, 1], dtype=int)
 test_label = np.empty([test_imgn * Classes, 1], dtype=int)
 test_data = np.empty([test_imgn * Classes, dimensions], dtype=float)
 random=randd.randint(0,42)
 i=0

 for x in range(0,Total,170):
   featur=fea[x:x+170]
   fea_tra,fea_test=train_test_split(featur,test_size=test_imgn,train_size=train_imgn,random_state=random)
   training_data[i*train_imgn:(i*train_imgn)+train_imgn]=fea_tra
   test_data[i*test_imgn:(i*test_imgn)+test_imgn]=fea_test
   #training_label[i*train_imgn:(i*train_imgn)+train_imgn]=int(gnd[x])
   #test_label[i*test_imgn:(i*test_imgn)+test_imgn]=int(gnd[x])
   training_label[i * train_imgn:(i * train_imgn) + train_imgn] = i+1
   test_label[i * test_imgn:(i * test_imgn) + test_imgn] = i+1
   i=i+1
 return training_data,training_label,test_data,test_label


# Computing Accuracy of Classifier
def Mainn(training_data, training_label,test_data,test_label,K,measure):
    S=time.time()
    x=0
    for i in range(len(test_data)):
        prediction = KNN(training_data, training_label, test_data[i], K,measure)
        if prediction == test_label[i]:
            x=x+1
    accuracy=(x/len(test_data))*100
    T=time.time()
    return accuracy,(T-S)
# Function for computing K nearest neighbour
def KNN(training_data,training_labels,test_data,k,measure):
    distance=[]
    if(measure=='Euclidean'):
         for i in range(len(training_data)):
             dist = np.sqrt(np.sum((training_data[i] - test_data) ** 2))
             distance.append(dist)
    if (measure == 'Mahalanobis'):
        inv_cov = np.linalg.inv(np.cov(training_data, rowvar=False))
        for i in range(len(training_data)):
            dist = np.matmul(np.matmul(np.transpose(training_data[i] - test_data), inv_cov),
                             (training_data[i] - test_data))
            #dist = da.mahalanobis(training_data[i], test_data, inv_cov)
            distance.append(dist)
    if (measure == 'Cosine Similarity'):
        for i in range(len(training_data)):
            dist = np.dot(training_data[i], test_data) / (np.linalg.norm(training_data[i]) * np.linalg.norm(test_data))
            distance.append(1-dist)
    #print(distance)
    indexes=np.argsort(distance)[:k]
    labels=training_labels[indexes]
    mode=scipy.stats.mode(labels)
    label=mode[0]
    return label
def PCA(X, new_dim):
    # Subtracting Mean from data
    X_meaned = X - np.mean(X, axis=0)
    # Finding Covariance Matrix
    cov_mat = np.cov(X_meaned, rowvar=False)
    # Finding eigen Vales and vectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    # Sorting those eigen values
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # Picking first (new dim) vectors as they are principal compenents
    eigenvector_subset = sorted_eigenvectors[:, 0:new_dim]
    # Transforming data
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    # Calcuting covariance matrix of data
    cov_mat_PCA = np.cov(X_reduced, rowvar=False)

    return X_reduced, cov_mat,cov_mat_PCA

a_euc,a_cos,a_mahal,t_euc,t_cos,t_mahal=[],[],[],[],[],[]
t0=time.time()
New_fea, covariance,covariance_PCA=PCA(fea,64)
t1=time.time()
xaxis=[]
for i in range(0,5):
    #calcaccuracy(fea,dim,train_size,testsize,total images,classes)
    training_data,training_label,test_data,test_label=Split(New_fea,64,100,70,1700,10)
    accuracy_euc,time_euc=Mainn(training_data,training_label,test_data,test_label,3,'Euclidean')
    accuracy_mahal,time_mahal = Mainn(training_data,training_label,test_data, test_label, 3, 'Mahalanobis')
    accuracy_cosine,time_cosine = Mainn(training_data,training_label,test_data, test_label, 3, 'Cosine Similarity')
    a_euc.append(accuracy_euc)
    a_mahal.append(accuracy_mahal)
    a_cos.append(accuracy_cosine)
    t_euc.append(time_euc)
    t_mahal.append(time_mahal)
    t_cos.append(time_cosine)
mean_euc,mean_cos,mean_mahal=np.mean(a_euc),np.mean(a_cos),np.mean(a_mahal)
end=time.time()
print('Time Taken for PCA',t1-t0)
print('Mean Time Taken for KNN using Euclidean',np.mean(t_euc))
print('Mean Time Taken for KNN using Mahalanobis',np.mean(t_mahal))
print('Mean Time Taken for KNN using Cosine Similarity',np.mean(t_cos))

print('Accuracy after random splits 5 time using Euclidean',a_euc)
print('Mean accuracy using Euclidean',mean_euc)
print('Std Deviation using Euclidean',np.std(a_euc))
print('Accuracy after random splits 5 time using Mahalanobis',a_mahal)
print('Mean accuracy using Mahalanobis',mean_mahal)
print('Std Deviation using Mahalanobis',np.std(a_mahal))
print('Accuracy after random splits 5 time using Cosine Similarity',a_cos)
print('Mean accuracy using Cosine Similarity',mean_cos)
print('Std Deviation using Cosine Similarity',np.std(a_cos))



x=[1,2,3,4,5]
plt.plot(x,a_euc,'g^',label='Accuracy using Euclidean')
plt.plot(x,a_mahal,'b^',label='Accuracy using Mahalanobis')
plt.plot(x,a_cos,'r^',label='Accuracy using Cosine Similarity')
plt.xlabel('Repetition')
plt.ylabel('Accuracy')
plt.title('Accuracy Plots')
plt.legend()
plt.grid()
plt.show()

plt.plot(x,t_euc,'g^',label='Time using Euclidean')
plt.plot(x,t_mahal,'b^',label='Time using Mahalanobis')
plt.plot(x,t_cos,'r^',label='Time using Cosine Similarity')
plt.xlabel('Repetition')
plt.ylabel('Time')
plt.title('Time Plots')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(9, 9))
plt.imshow(covariance, cmap='gray');
plt.title('Covariance Matrix before PCA')
plt.show()
# Now plotting Covariance Matrix after Dimensionality reduction
plt.imshow(covariance_PCA, cmap='gray');
plt.title('Covariance Matrix after PCA')
plt.show()


k_size=[]
a_euc,a_cos,a_mahal,t_euc,t_cos,t_mahal=[],[],[],[],[],[]
training_data,training_label,test_data,test_label=Split(New_fea,64,100,70,1700,10)
for i in range(1,30,3):
    k_size.append(i)
    #calcaccuracy(fea,dim,train_size,testsize,total images,classes,randoms)

    accuracy_euc,time_euc=Mainn(training_data,training_label,test_data,test_label,i,'Euclidean')
    #accuracy_mahal,time_mahal = Mainn(training_data,training_label,test_data, test_label, i, 'Mahalanobis')
    accuracy_cosine,time_cosine = Mainn(training_data,training_label,test_data, test_label, i, 'Cosine Similarity')
    a_euc.append(accuracy_euc)
    a_mahal.append(accuracy_mahal)
    a_cos.append(accuracy_cosine)
    t_euc.append(time_euc)
    t_mahal.append(time_mahal)
    t_cos.append(time_cosine)
mean_euc,mean_cos,mean_mahal=np.mean(a_euc),np.mean(a_cos),np.mean(a_mahal)
end=time.time()
print(a_euc)


plt.plot(k_size,a_euc,'g^',label='Accuracy using Euclidean')
plt.plot(k_size,a_mahal,'b^',label='Accuracy using Mahalanobis')
plt.plot(k_size,a_cos,'r^',label='Accuracy using Cosine Similarity')
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.title('Accuracy Plots')
plt.legend()
plt.grid()
plt.show()

plt.plot(k_size,t_euc,'g^',label='Time using Euclidean')
plt.plot(k_size,t_mahal,'b^',label='Time using Mahalanobis')
plt.plot(k_size,t_cos,'r^',label='Time using Cosine Similarity')
plt.xlabel('Value of K')
plt.ylabel('Time')
plt.title('Time Plots')
plt.legend()
plt.grid()
plt.show()

plt.plot(covariance)
plt.show