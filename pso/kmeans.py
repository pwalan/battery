from numpy import *


# calculate Euclidean distance

def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
# dataSet is a matrix
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)

    print('Congratulations, cluster complete!')
    return centroids, clusterAssment


if __name__ == '__main__':
    dataSet = [[1, 1, 1],
               [1, 2, 1],
               [1, 3, 1],
               [2, 1, 1],
               [2, 2, 1],
               [2, 3, 1],
               [3, 1, 1],
               [3, 2, 1],
               [3, 3, 1], ]
    dataSet = mat(dataSet)
    k = 3
    centroids, clusterAssment = kmeans(dataSet, k)
    print(centroids)
    print(clusterAssment.tolist()[0][0] == 0,
          clusterAssment.tolist()[0][0] == 1,
          clusterAssment.tolist()[0][0] == 2, )
