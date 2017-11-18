import numpy as np
import time
from sklearn.utils import safe_indexing

tic = time.time()

data = np.genfromtxt('train_data.csv', delimiter=',',dtype=int)
y = data[:,-1]
X = data[:,:-1]

X_test = np.genfromtxt('test_data.csv', delimiter=',',dtype=int)
y_test = np.genfromtxt('test_hand.csv', delimiter=',',dtype=int)

def oversample(X,y):
	unq, target_stats = np.unique(y, return_counts=True)
	max_number = np.max(target_stats)
	sample_indices = range(X.shape[0])
	for class_sample in sorted(unq): 
		num_samples = max_number - target_stats[class_sample]
		num_samples = int(num_samples)
		target_class_indices = np.flatnonzero(y == class_sample)
		indices = np.random.randint(low=0, high=target_stats[class_sample], size=num_samples)
		sample_indices = np.append(sample_indices,target_class_indices[indices])
	return (safe_indexing(X, sample_indices),safe_indexing(y, sample_indices))

def transform(X):
        N = X.shape[0]
        new_X = np.zeros((N,52),dtype=int)
        for i in range(N):
                for j in range(5):
                        m = 2*j
                        n = 2*j + 1
                        index = (X[i,m]-1)*13 + X[i,n] - 1
                        new_X[i,index] = 1
        return new_X



#############################################################
############## TASK 1 STARTS HERE ###########################
#############################################################
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
clf = MLPClassifier(hidden_layer_sizes = [100,100,100,100,100],verbose=1,learning_rate='adaptive' )


X_train = transform(X)
X_train, y = oversample(X_train, y)
print('Shape of the Oversampled Data', X_train.shape)
X_test_transform = transform(X_test)

clf.fit(X_train,y)
print('Train accuracy %f' %clf.score(X_train,y))
print('Test accuracy %f' %clf.score(X_test_transform,y_test))
a = clf.predict(X_test_transform)
a = a.astype(int)
np.savetxt("output_task1.csv", a, fmt='%d')

#############################################################
######  TASK 2 STARTS HERE   ################################
#############################################################
N = X_test.shape[0]
X_new = []
for i in range(N):
        hand = []
        test = []
        for j in range(5):
                hand.append((X[i,2*j]-1)*13 + X[i,2*j+1])
        hand = np.array(hand)
        for j in range(5):
                for k in range(52):
                        z = k+1
                        if (np.sum(hand==z)==0):
                                #n=z/13 + 1
                                n=z//13 + 1
                                m=z%13
                                if m==0:
                                        n -= 1
                                        m = 13
                                a=list(X_test[i])
                                a[2*j] = n
                                a[2*j+1] = m
                                test.append(a)
        test = np.array(test)
        test_transform = transform(test)
        scores = clf.predict(test_transform)
        max_hand = np.max(scores)
        init_hand = clf.predict(np.reshape(X_test_transform[i],(1,-1)))
        if max_hand > init_hand:
                X_new.append(test[np.argmax(scores)])
        else:
                X_new.append(X_test[i])

X_new = np.array(X_new)
X_new = X_new.astype(int)
np.savetxt("output_task2.csv",X_new,fmt='%d',delimiter=',')

toc = time.time()
print('Total time elapsed is: %.1f minutes' %((toc-tic)/60))
