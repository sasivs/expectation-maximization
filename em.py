import numpy as np, math, pandas as pd
from scipy.stats import multivariate_normal
from decimal import Decimal


def clustering(points):
    points = list(set(points)) #To eliminate reduntant points
    import random
    '''
    Initialize random gaussians!
    '''
    mean = [np.array(list(points[random.randint(0,len(points)-1)])), np.array(list(points[random.randint(0,len(points)-1)])), np.array(list(points[random.randint(0,len(points)-1)]))]
    covariance1 = np.cov(np.transpose(points))
    weights = np.array([0.3, 0.3, 0.4])
    covariance = [covariance1, covariance1, covariance1]
    init_log_likelihood = None #Log likelihood to test convergence None initially
    threshold = 0.01 #Threshold difference for convergence

    while True:

        prob_dict = {point:[] for point in points}
        new_means = [np.array([0,0,0,0]) for _ in range(3)]
        new_covariance = [np.array([[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]) for _ in range(3)]

        log_likelihood = Decimal(0) #Using decimal to increase the precision of floating point numbers

        for point in points:

            prob_sum = 0 #Prob sum for normalization

            for _ in range(3):
                prob =  multivariate_normal(mean=mean[_], cov=covariance[_]).pdf(point) #Finding gaussian probability
                prob = prob*weights[_] #Multiply gaussian probability with weight
                prob_dict[point].append(prob)
                prob_sum += prob
            
            prob_sum = Decimal(prob_sum)

            log_likelihood += prob_sum.ln() #comprehensive computation of log likelihood

            for _ in range(3):
                prob_dict[point][_] = prob_dict[point][_]/float(prob_sum) #Normalizing the probability
                new_means[_] = new_means[_]+np.dot(prob_dict[point][_], list(point)) #New means for the next iteration

        for _ in range(3):
            weights[_] = sum([prob_item[1][_] for prob_item in prob_dict.items()]) #New weights
            new_means[_] = np.divide(new_means[_], weights[_])
            
        for _ in range(3):
            prob_sum = sum([prob_item[1][_] for prob_item in prob_dict.items()])
            new_covariance[_] = np.cov(np.transpose(points), bias=True, aweights=[prob_item[1][_]/prob_sum for prob_item in prob_dict.items()])
            weights[_] = weights[_]/len(points)     

        mean = new_means 
        covariance = new_covariance

        #check for convergence
        if init_log_likelihood and abs(init_log_likelihood-log_likelihood)<=threshold:
            break

        init_log_likelihood = log_likelihood

    probabilities = np.zeros(shape=(len(points),3))
    for i in range(3):
        distribution = multivariate_normal(mean=mean[i], cov=covariance[i])
        probabilities[:,i] = distribution.pdf(points)
    numerator = probabilities*np.array([weights])
    denominator = numerator.sum(axis=1)[:, np.newaxis]
    probabilities = numerator/denominator

    print("Means: ", mean)
    print("Covariance: ", covariance)
    print("Weights: ", weights)
    print("Log likelihood: ", log_likelihood)

    return np.argmax(probabilities, axis=1)


def get_points():
    '''
    Read points from csv file.
    '''
    file = pd.read_csv("D:\\Documents\\ai_sem5\\em_for_dataset\\dataset.csv")
    points = []
    for _,row in file.iterrows():
        points.append((row['A'], row['B'], row['C'], row['D']))
    # file.close()
    return points
    
def main():
    points = get_points()
    assignment = clustering(points)
    print(assignment)
if __name__ == '__main__':
    main()
