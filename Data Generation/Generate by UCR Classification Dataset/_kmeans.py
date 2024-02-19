import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from _sbd import _sbd


# Calculated Euclidean distance
def calcDis(dataSet, centroids, k, method):
    if method == 'euclid':
        clalist=[]
        for data in dataSet:
            diff = np.tile(data, (k, 1)) - centroids  
            squaredDiff = diff ** 2     
            squaredDist = np.sum(squaredDiff, axis=1)   
            distance = squaredDist ** 0.5  
            clalist.append(distance) 
        # clalist = np.array(clalist) 
        return clalist  
    elif method == 'sbd':
        clalist = []
        for data_ in dataSet:
            # print('data: ', data_)
            dis_ = []
            for cent_ in centroids:
                # print('cent: ', cent_)
                dis_.append(_sbd(data_,cent_)[0])
                # print('dis: ', dis_)
            dis_ = np.array(dis_)
            clalist.append(dis_)
        # clalist = np.array(clalist)
        # print(clalist)
        return clalist
    else:
        print('Error! Please enter a correct distance.')

# Computed center of mass
def classify(dataSet, centroids, k):
    # Calculate the distance from samples to centroids
    clalist = calcDis(dataSet, centroids, k, method='sbd')
    # Group and calculate new centroids
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 means to find the index of the minimum value in each row
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean()  # DataFrame(dataSet) groups DataSet, groupby(min) categorizes by min, mean() calculates the mean of the categorized results
    newCentroids = newCentroids.values
    centroids = np.array(centroids)
    # Calculate the change
    changed = newCentroids - centroids
    return changed, newCentroids

# Using k-means clustering
def kmeans(dataSet, k=2 , tol = 1e-10, max_step = 100, method = 'sbd', n_iter = 10, init = 'k-means++', ratio_abandoned = 0.05 ): # 

    if type(dataSet)==np.ndarray:
        dataSet = dataSet.tolist()

    centroids_ls = []
    cluster_ls = []
    label_ls = []
    shortest_dis_sum_ls = [] 
    
    for iterr_ in range(n_iter):
        # Randomly select centroids
        if init == 'random':
            centroids = random.sample(dataSet, k)
        # print(centroids)
        if init == 'k-means++':
            centroids_first = random.sample(dataSet, 1)
            centroids_select = centroids_first
            # print(centroids_select)
            for i in range(1,k):
                all_distances = np.empty((len(dataSet),0))
                for j in centroids_select:
                    distances = np.array(calcDis(dataSet,[j],k,method='sbd')).reshape(-1,1)
                    all_distances = np.c_[all_distances,distances]
                    # print(distances)
                    # print(all_distances)
                min_distances = all_distances.min(axis=1).reshape(-1,1)
                index = np.argmax(min_distances)
                centroids_select.append(dataSet[index])
                # print(dataSet[index])
            centroids = centroids_select
        else:
            print("Error!")
        
        # Update centroids until changes are zero
        changed, newCentroids = classify(dataSet, centroids, k)
        step_ = 0 
        while np.any(changed != 0):
            changed, newCentroids = classify(dataSet, newCentroids, k)
            if np.max(np.abs(changed)) < tol:
                # print('1')
                break
            if step_ >= max_step:
                # print('2')
                break
            step_ = step_ + 1

        centroids = sorted(newCentroids.tolist())  # Convert matrix to list and sort
        # Calculate clusters based on centroids
        cluster = []
        label = []
        clalist = calcDis(dataSet, centroids, k, method='sbd')  # Call Euclidean distance
        clalist = np.array(clalist)
        clalist_min = np.min(clalist, axis=1)
        abandon_list = np.argsort(clalist_min)[-int(ratio_abandoned * len(clalist_min)) - 1:]
        minDistIndices = np.argmin(clalist, axis=1)
        shortest_dis = []
        for (idx_, value_) in enumerate(minDistIndices):
            shortest_dis.append(np.array(clalist)[idx_,value_])
        shortest_dis_sum = np.sum(np.array(shortest_dis)**2)

        for i in range(k):
            cluster.append([])
        for i, j in enumerate(minDistIndices):  
            cluster[j].append(dataSet[i])
            if i in abandon_list:
                label.append(-1)
            else:
                label.append(j)
            
        centroids_ls.append(centroids)
        cluster_ls.append(cluster)
        label_ls.append(label)
        shortest_dis_sum_ls.append(shortest_dis_sum)

    return centroids_ls[shortest_dis_sum_ls.index(min(shortest_dis_sum_ls))], cluster_ls[shortest_dis_sum_ls.index(min(shortest_dis_sum_ls))], label_ls[shortest_dis_sum_ls.index(min(shortest_dis_sum_ls))]
 
# Creating demo Data
def createDataSet():
    return [[1, 1, 1], [1, 2, 1], [2, 1, 0], [6, 4, 5], [6, 3, 4], [5, 4, 6]]

if __name__=='__main__': 
    dataset = createDataSet()
    dataset = np.random.rand(40,2)

    # quit()
    # dataset = np.array(dataset)
    # print(dataset)
    centroids, cluster, label = kmeans(dataset, 3)
    print('Center of mass: %s' % centroids)
    print('Cluster: %s' % cluster)
    print('Labels: %s' % label)

    for i in range(len(dataset)):
        if label[i] == 0:
            plt.scatter(dataset[i][0],dataset[i][1], marker = 'o',color = 'green', s = 40 ,label = 'Original Points')
        if label[i] == 1:
            plt.scatter(dataset[i][0],dataset[i][1], marker = 'o',color = 'orange', s = 40 ,label = 'Original Points')
        if label[i] == 2:
            plt.scatter(dataset[i][0],dataset[i][1], marker = 'o',color = 'grey', s = 40 ,label = 'Original Points')
    for j in range(len(centroids)):
        plt.scatter(centroids[j][0],centroids[j][1],marker='x',color='red',s=50,label='center of mass')
    plt.show()
