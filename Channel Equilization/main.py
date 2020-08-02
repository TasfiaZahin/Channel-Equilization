import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import math
import operator
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import time
import itertools

np.set_printoptions(suppress=True)

trans_prob = 0.5
#trans_prob2 = 0.5
l = 2

def getGaussian(xvect, mewk, sigmak):

    xi = np.zeros((l,1))
    for i in range(len(xi)):
        xi[i][0] = xvect[i]
    #print(xi)

    det_sigmak = np.linalg.det(sigmak)
    det_sigmak = np.absolute(det_sigmak)
    # print("sigmak",sigmak)
    # print("det_sigmak",det_sigmak)

    if(det_sigmak == 0):
        print("determinant zero")
        det_sigmak = 0.0000001

    inv_sigmak = np.linalg.inv(sigmak)
    constant = 1.0 / (np.sqrt((np.power(2*np.pi,l)) * det_sigmak))
    #print("deno const",np.sqrt((np.power(2*np.pi,dim)) * det_sigmak))
    #print("constant:", constant)

    #print("xi", xi)
    #print("mewk", mewk)

    xi_min_mewk = np.subtract(xi,mewk)

    # print(xi_min_mewk,"\nhh",inv_sigmak)

    #print("trans",np.transpose(xi_min_mewk))
    #print("inve sigmak", inv_sigmak)
    temp = np.dot(np.transpose(xi_min_mewk),inv_sigmak)
    #print("temp",temp)
    #print("ximinmewk", xi_min_mewk)
    exp_val = np.dot(temp,xi_min_mewk)
    exp_val = -0.5 * exp_val
    #print("exp val:",exp_val)

    #print("sigmak",sigmak)

    #print(np.multiply(5,mewk))

    if(exp_val < -500):
        exp_val = -500
    elif (exp_val > 500):
        exp_val = 500

    ans = constant*np.exp(exp_val)
    #print("gaussian",ans)
    return ans

def getClusterIndex(cluster_dict, lst):
    # print(cluster_dict)
    idx = -1
    for key, val in cluster_dict.items():
        # print("val",val)
        # print("lst",lst)
        if val == lst:
            idx = key
    # print(idx)
    # idx = int(idx)
    return idx

def getTransitions(cluster_dict):

    trans_prob_dict = {}
    for i in range(len(cluster_dict)):
        lst = cluster_dict[i].copy()
        del lst[0]

        pos1 = lst.copy()
        pos1.append(0)
        pos2 = lst.copy()
        pos2.append(1)

        # print(lst)
        # print(pos1)
        # print(pos2)

        idx1 = getClusterIndex(cluster_dict,pos1)
        idx2 = getClusterIndex(cluster_dict,pos2)

        new_list = []
        new_list.append(idx1)
        new_list.append(idx2)

        trans_prob_dict[i] = new_list
        #print("newlist:",new_list)
        #print()

    #print("transitions:",trans_prob_dict)
    return  trans_prob_dict

def getDist(xi,mewk):

    # print("xi",xi)
    # print("mewk",mewk)
    sum = 0.0
    for i in range(len(xi)):
        #print(xi[i],mewk[i])
        sum += (xi[i]-mewk[i])**2
    return np.sqrt(sum)

class Cluster:

    def __init__(self,l):
        self.count = 0.0
        self.label = None
        self.x_vect = []
        self.mean = np.zeros((l,1))
        self.covar = np.zeros((l,l))

class Node:

    def __init__(self,num):
        self.number = num
        self.cost = -1
        self.prev = -1

class Channel:

    def __init__(self,n,h,nk_mew,nk_var):
        self.n = n
        self.l = l
        self.h = h
        self.comb_len = n + l - 1
        self.nk_mew = nk_mew
        self.nk_var = nk_var

        self.cluster_dict = {}
        self.clusters = {}
        self.network = {}

        lst = list(map(list, itertools.product([0, 1], repeat=self.comb_len)))
        # print(lst)

        for i in range(np.power(2,self.comb_len)):
            self.cluster_dict[i] = lst[i]
            self.clusters[i] = Cluster(self.l)

        self.transition_dict = getTransitions(self.cluster_dict)

        #print(cluster_dict)
        #print(clusters)
        #getClusterIndex(cluster_dict,[0,1,0])

    def getNoise(self):
        ans = np.random.normal(self.nk_mew, np.sqrt(self.nk_var))
        return ans

    def setClusterInfo(self):

        for i in range(len(self.clusters)):
            clst = i
            cluster = self.clusters[i]
            cluster.x_vect = np.array(cluster.x_vect)

            if(len(cluster.x_vect) > 0):

                mew = np.mean(cluster.x_vect, axis=0)
                #print(mew)

                for k in range(self.l):
                    cluster.mean[k] = mew[k]
            #print("mean:\n",cluster.mean)

            if (len(cluster.x_vect) > 1):

                covar = np.cov(np.transpose(cluster.x_vect))
                #print(covar)
                cluster.covar = covar
            else:
                print("finding covar sample zero or one!")

            #print("covar:\n",cluster.covar)

        for i in range(len(self.clusters)):
            print("cluster ",i)
            cluster = self.clusters[i]
            print("mean:\n", cluster.mean)
            print("covar:\n", cluster.covar)

    def processTrain(self, bitString):

        #pad comb_len-1 number of zeroes at end of string, so new length will be (len of string + comb_len - 1)

        bitString = bitString[::-1]
        #print(bitString)
        bitString = bitString.zfill(len(bitString) + self.comb_len-1)
        #print(bitString)
        bitString = bitString[::-1]
        #print(bitString)

        #take comb_len amnt of subsets to form matrix from bit String, store xk values acc to cluster

        for i in range (len(bitString) - 1, self.comb_len - 2, -1): #back to front

            lst = []
            for j in range(-(self.comb_len-1), 1, 1):
                lst.append(float(bitString[i+j]))

            clst = getClusterIndex(self.cluster_dict,lst)

            cluster = self.clusters[clst]
            cluster.label = lst[0]
            xvect = cluster.x_vect

            indv_vect = []
            for j in range(self.l):
                sum = 0.0
                # print("hi",j)
                for k in range(self.n):
                    sum += lst[j + k] * self.h[k]
                sum += self.getNoise()  # adding noise
                indv_vect.append(sum)

            xvect.append(indv_vect)
            cluster.count +=1

        # for i in range(len(self.clusters)):
        #     cluster = self.clusters[i]
        #     print("cluster number ", i)
        #     print("cluster label ",cluster.label)
        #     print("x_vect:\n",np.array(cluster.x_vect))

        #set mean variance of clusters
        self.setClusterInfo()


        #print(self.transition_dict)

    def processTest(self, bitString):

        bitString = bitString[::-1]
        bitString = bitString.zfill(len(bitString) + self.comb_len - 1)
        bitString = bitString[::-1]

        xvect = []
        # take comb_len amnt of subsets to form matrix from bit String, get xk values
        for i in range (len(bitString) - 1, self.comb_len - 2, -1): #back to front

            lst = []
            for j in range(-(self.comb_len-1), 1, 1):
                lst.append(float(bitString[i+j]))

            indv_vect = []
            for j in range(self.l):
                sum = 0.0
                # print("hi",j)
                for k in range(self.n):
                    sum += lst[j + k] * self.h[k]
                sum += self.getNoise()  # adding noise
                indv_vect.append(sum)

            xvect.append(indv_vect)

        xvect = np.array(xvect)
        #print("test x vectors:\n",xvect)

        return xvect

    def getMean(self,clst):
        return self.clusters[clst].mean

    def getCovar(self,clst):
        return self.clusters[clst].covar

    def getPrior(self,clst):
        total = 0.0
        for i in range(len(self.clusters)):
            total += self.clusters[i].count

        ans = self.clusters[clst].count/total
        return ans

    def getGaussianProb(self,xi,w):

        cluster = self.clusters[w]
        mewk = cluster.mean
        sigmak = cluster.covar
        return getGaussian(xi,mewk,sigmak)


    #def getTransProb(self,w1,w2):


    def runViterbi(self, xvect):
        #initialize network to run viterbi
        lenlayers = len(xvect)
        #print(lenlayers)

        for i in range(lenlayers):
            length = len(self.clusters)
            nodes = []
            for j in range(length):
                nodes.append(Node(j))
            self.network[i] = nodes
        #print(self.network)

        #run viterbi

        #first layer set cost p(wi)*prob of x0 given class
        first_layer = self.network[0]
        #print(len(first_layer))

        for j in range(len(first_layer)):
            node = first_layer[j]
            #print(node.number)
            node.cost = np.log(self.getPrior(node.number)) + np.log(self.getGaussianProb(xvect[0],node.number))
            #print(node.cost)


        #for rest of the layers
        for i in range(1,lenlayers):

            layer = self.network[i]
            prev_layer = self.network[i-1]

            for j in range(len(layer)):

                node = layer[j]
                xgivenw = self.getGaussianProb(xvect[i],node.number)

                prev_nodes = self.transition_dict[node.number]
                #print(prev_nodes)

                cost_array = []
                for k in range(len(prev_nodes)):
                    cost = prev_layer[prev_nodes[k]].cost + np.log(trans_prob) + np.log(xgivenw)
                    cost_array.append(cost)
                #print(cost_array)
                best_cost = np.max(cost_array)
                cost_idx = np.argmax(cost_array)

                best_idx = prev_nodes[int(cost_idx)]
                #print(best_cost, " best index", best_idx)
                node.cost = best_cost
                node.prev = best_idx

                # idx1 = prev_nodes[0]
                # idx2 = prev_nodes[1]
                #
                # new_cost1 = prev_layer[idx1].cost + np.log(trans_prob) + np.log(xgivenw)
                # new_cost2 = prev_layer[idx2].cost + np.log(trans_prob) + np.log(xgivenw)
                #
                # print("new costs: ",new_cost1,";",new_cost2)
                #
                # if(new_cost1 > new_cost2):
                #     node.cost = new_cost1
                #     node.prev = idx1
                # else:
                #     node.cost = new_cost2
                #     node.prev = idx2
                # print(node.cost,node.prev)

        # for i in range(lenlayers):
        #     print("layer",i)
        #     layer = self.network[i]
        #     for j in range(len(layer)):
        #         node = layer[j]
        #         print("node",j,"; cost:",node.cost,"; prev:",node.prev)
        #     print()

        lst = []
        last_layer = self.network[lenlayers-1]

        for j in range(len(last_layer)):
            node = last_layer[j]
            lst.append(node.cost)
        #print("last layer costs: ",lst)
        best_idx = np.argmax(lst)
        #print(best_idx)

        bits = []
        #backtracking
        node = last_layer[best_idx]
        bits.append(self.clusters[node.number].label)
        #print(self.clusters[node.number].label)

        for i in range(lenlayers-2,-1,-1):
            prev_layer = self.network[i]
            node = prev_layer[node.prev]
            bits.append(self.clusters[node.number].label)
            #print(self.clusters[node.number].label)

        print("output bits:", bits)
        # print(len(bits))
        # print(len(xvect))

        return bits

    def runViterbiDist(self, xvect):

        #initialize network to run viterbi
        lenlayers = len(xvect)
        #print(lenlayers)

        for i in range(lenlayers):
            length = len(self.clusters)
            nodes = []
            for j in range(length):
                nodes.append(Node(j))
            self.network[i] = nodes
        #print(self.network)

        #run viterbi

        #first layer set cost p(wi)*prob of x0 given class
        first_layer = self.network[0]
        #print(len(first_layer))

        for j in range(len(first_layer)):
            node = first_layer[j]
            #print(node.number)
            node.cost = getDist(xvect[0],self.clusters[j].mean)
            #print(node.cost)


        #for rest of the layers
        for i in range(1,lenlayers):

            layer = self.network[i]
            prev_layer = self.network[i-1]

            for j in range(len(layer)):

                node = layer[j]
                #xgivenw = self.getGaussian(xvect[i],node.number)

                prev_nodes = self.transition_dict[node.number]
                #print(prev_nodes)

                cost_array = []
                for k in range(len(prev_nodes)):
                    cost = prev_layer[prev_nodes[k]].cost + getDist(xvect[i],self.clusters[j].mean)
                    cost_array.append(cost)
                #print(cost_array)
                best_cost = np.min(cost_array)
                cost_idx = np.argmin(cost_array)

                best_idx = prev_nodes[int(cost_idx)]
                #print("prbbbb,",cost_idx)
                #print(best_cost, " b", best_idx)
                node.cost = best_cost
                node.prev = best_idx

                # idx1 = prev_nodes[0]
                # idx2 = prev_nodes[1]
                #
                # new_cost1 = prev_layer[idx1].cost + getDist(xvect[i],self.clusters[j].mean)
                # new_cost2 = prev_layer[idx2].cost + getDist(xvect[i],self.clusters[j].mean)
                #
                # #print("new costs: ",new_cost1,";",new_cost2)
                #
                # if(new_cost1 < new_cost2):
                #     node.cost = new_cost1
                #     node.prev = idx1
                # else:
                #     node.cost = new_cost2
                #     node.prev = idx2
                # print(node.cost, node.prev)

        # for i in range(lenlayers):
        #     print("layer",i)
        #     layer = self.network[i]
        #     for j in range(len(layer)):
        #         node = layer[j]
        #         print("node",j,"; cost:",node.cost,"; prev:",node.prev)
        #     print()

        lst = []
        last_layer = self.network[lenlayers-1]

        for j in range(len(last_layer)):
            node = last_layer[j]
            lst.append(node.cost)
        #print("last layer costs: ",lst)
        best_idx = np.argmin(lst)
        #print(best_idx)

        bits = []
        #backtracking
        node = last_layer[best_idx]
        bits.append(self.clusters[node.number].label)
        #print(self.clusters[node.number].label)

        for i in range(lenlayers-2,-1,-1):
            prev_layer = self.network[i]
            node = prev_layer[node.prev]
            bits.append(self.clusters[node.number].label)
            #print(self.clusters[node.number].label)

        print("output bits:",bits)
        # print(len(bits))
        # print(len(xvect))

        return bits


def main():

    train_string = open('train.txt', 'r').read()
    #print(len(train_string))

    test_string = open('test.txt', 'r').read()

    n = 3
    h = [10, 15, 20]  # change n and h both for diff
    #comb_len = 2 * l - 1
    nk_mew = 2
    nk_var = 0.6

    #np.random.seed(1)
    print("Processing...")

    start = time.time()

    ch = Channel(n,h,nk_mew,nk_var)
    ch.processTrain(train_string)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    x_vect = ch.processTest(test_string)
    bits = ch.runViterbi(x_vect)
    #bits = ch.runViterbiDist(x_vect)

    with open('out.txt', 'w') as f:
        for item in bits:
            f.write("%s" % int(item))

    total = len(test_string)
    count_error = 0
    for i in range(len(test_string)):
        #print("actual:",int(test_string[i]))
        #print("verdict:",int(bits[i]))
        if(int(bits[i]) != int(test_string[i])):
            count_error += 1

    acc = (total-count_error)/total*100
    print("acc: ",acc,"%")
    print("Elapsed training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    # sample()

    # nk = getNoise(0,0.1)
    # print(nk)

main()