import random
import numpy as np
import math

"""
Function that computes the Euclidean distance
betwen two points (x1,y1) and (x2,y2).
"""


def distance(x1, y1, x2, y2):
    d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return d


"""
Ordinary implementation of Dijkstra's Algorithm.
G[u][v]=w implies a connection from u->v with
a cost/weight of w.
Inputs:
    G, s, t
    graph, start vertex, terminal vertex
Output:
    length of shortest path from s->t
"""


def Dijkstra(G, s, t):
    dist = {}  # dist[i] is the shortest known distance from s to i.
    prev = {}  # prev[i]=j means that vertex j immediately precedes vertex i on the shortest known path from s to i..
    visited = []  # set of vertices that have already appeared at the top of the Q and had all neighbors explored.  these vertices can be excluded from future queues.
    Q = G.keys()  # initially all vertices may be part of the queue.

    for vertex in Q:  # initializing shortest known distances.
        dist[vertex] = float("inf")
        prev[vertex] = None
    dist[s] = 0

    while (t not in visited):
        X = sorted(dist,
                   key=dist.get)  # sorts vertices according to value of dist.  (i.e. if dist[i] < dist[j], then i comes before j in the list X.)
        Q = []
        for x in X:
            if (x not in visited):
                Q.append(x)
        currVertex = Q[0]
        # print(G[currVertex])
        for neighbor in G[currVertex].keys():
            if (dist[neighbor] > dist[currVertex] + G[currVertex][neighbor]):
                dist[neighbor] = dist[currVertex] + G[currVertex][neighbor]
                prev[neighbor] = currVertex
        visited.append(currVertex)
    # print('Tracing back the line of predecessors from the destination:')
    cur = t  # starting at the destination
    path = []
    while (cur != s):  # and until we get to the start location
        # print('Vertex with index:')
        # print(cur)
        path.append(cur)
        cur = prev[cur]  # move from current vertex to predecessor.
    path.append(s)
    path.reverse()
    return dist[t]


"""Code block for generating network data."""
random.seed(1)
xCoord = []
yCoord = []
origRate = []
destRate = []
driverDensity = []
avgFareMultiplier = []

"""
Feel free to modify the following parameters while testing: radial, bFare, bDriverDensity, randDriverDensity, avgHourlyFare
"""
radial = False
uniformRandom = not radial
bFare = 5  # base fare
bDriverDensity = 1.5  # base driver density
randDriverDensity = 1.5  # random component of driver density
avgHourlyFare = 15

"""
generates a ring-shaped instance
with centered at location (0.5, 0.5)
"""
useEuclidDist = True
degree = {}
if (radial):
    numRings = 5  # number of rings
    verticesPerRing = 7  # number of vertices per ring
    numVertices = numRings * verticesPerRing
    for i in range(numRings):
        distFromCenter = 0.5 * (i + 1) / float(numRings)
        for j in range(verticesPerRing):
            degree[len(xCoord)] = 0
            angleFromCenter = 2 * j * math.pi / verticesPerRing
            xCoord.append(0.5 + distFromCenter * math.cos(angleFromCenter))
            yCoord.append(0.5 + distFromCenter * math.sin(angleFromCenter))
            origRate.append(1.2 / (0.3 + distance(0.5, 0.5, xCoord[len(xCoord) - 1], yCoord[len(xCoord) - 1])))
            destRate.append(random.random())
            driverDensity.append(bDriverDensity + randDriverDensity * random.random())
            avgFareMultiplier.append(2 * avgHourlyFare * random.random())  # $15/hour avg.
    destRate = np.array(destRate) / np.sum(destRate)
    G = {}
    for i in range(numVertices):
        G[i] = {}
        G[i][i] = 0
    for i in range(numVertices):
        for j in range(numVertices):
            if (abs(i - j) == verticesPerRing or (i / verticesPerRing == j / verticesPerRing and (
                    abs(i - j) == 1 or abs(i - j) == (verticesPerRing - 1)))):  # or
                d = distance(xCoord[i], yCoord[i], xCoord[j], yCoord[j])
                G[i][j] = d  # edge distances are measured in hours.

if (uniformRandom):
    numVertices = 100
    # Generating one-hundred random locations.
    xCoord = []
    yCoord = []
    for i in range(numVertices):
        xCoord.append(random.random())
        yCoord.append(random.random())
        origRate.append(1.2 / (0.3 + distance(0.5, 0.5, xCoord[len(xCoord) - 1], yCoord[len(xCoord) - 1])))
        destRate.append(random.random())
        driverDensity.append(bDriverDensity + randDriverDensity * random.random())
        avgFareMultiplier.append(2 * avgHourlyFare * random.random())  # $15/hour avg.

    # To randomly generate the graph,
    # I've arbitrarily chosen a rule
    # that any vertices within distance
    # 0.15 of one another will be connected
    # with an edge.
    G = {}
    minIdx = {}
    for i in range(numVertices):
        G[i] = {}
    for i in range(numVertices):
        minDist = 9999
        for j in range(numVertices):
            d = distance(xCoord[i], yCoord[i], xCoord[j], yCoord[j])
            if (d < minDist and j != i):
                minDist = d
                minIdx[i] = j
            if (d < 0.20):
                connectVertices = True
            else:
                connectVertices = False
            if (connectVertices):
                G[i][j] = d
        if (len(G[i]) == 1):  # if vertex i is connected only to self
            G[i][minIdx[i]] = minDist  # connect to the nearest vertex
            G[minIdx[i]][i] = minDist

import matplotlib.pyplot as plt

"""
Choose one of the four below to plot.
Only one value below should be True.
Size of vertices plotted scales with values.
"""
plotOriginRate = False
plotDestRate = False
plotDriverDensity = False
plotAvgFareMultiplier = True

if (useEuclidDist):
    plt.figure(figsize=(8, 8))
    if (plotOriginRate):
        plt.title("Distribution of ride origins")
        pOrigRate = 2000 * (np.array(origRate)) / np.sum(origRate)
        plt.scatter(xCoord, yCoord, s=pOrigRate)
    elif (plotDestRate):
        plt.title("Distribution of ride destinations")
        pDestRate = 2000 * np.array(destRate) / np.sum(destRate)
        plt.scatter(xCoord, yCoord, s=pDestRate)
    elif (plotDriverDensity):
        plt.title("Distribution of driver density")
        pDD = 2000 * np.array(driverDensity) / np.sum(driverDensity)
        plt.scatter(xCoord, yCoord, s=pDD)
    elif (plotAvgFareMultiplier):
        plt.title("Distribution of average fare multiplier")
        pAFM = 2000 * np.array(avgFareMultiplier) / np.sum(avgFareMultiplier)
        plt.scatter(xCoord, yCoord, s=pAFM)
    else:
        plt.scatter(xCoord, yCoord)
    for i in range(len(xCoord)):
        plt.annotate(str(i), (xCoord[i], yCoord[i] + 0.03))
    ax = plt.axes()
    for i in range(numVertices):  # for each vertex i
        for j in range(i + 1, numVertices):  # and vertex j
            if (j in G[i]):  # if i and j are connected
                if (G[i][j] < 999):
                    x = xCoord[i]
                    y = yCoord[i]
                    dx = xCoord[j] - xCoord[i]
                    dy = yCoord[j] - yCoord[i]
                    ax.arrow(x, y, dx, dy)  # plot an arrow connecting them

# plt.show()

"""generate instances"""
numToGenerate = 5  # number of instances to generate
lengthOfDrivingSession = 8  # length of driving session in hours

for i in range(numToGenerate):
    reqs = {}
    time = 0
    totalOrigRate = sum(origRate)
    fileName = "requestInstance" + str(i + 1) + ".csv"
    f = open(fileName, 'w')
    while (time < lengthOfDrivingSession):
        randNum = random.random()  # represents percentile survival

        # e^{-totalOrigRate*t} = randNum
        # -totalOrigRate*t = log(randNum)
        # t = log(randNum)/(-totalOrigRate)
        interarrivalTime = math.log(randNum) / (-totalOrigRate)
        time = time + interarrivalTime
        # print(time)
        if (time > lengthOfDrivingSession):
            break
        origProbability = np.array(origRate) / np.sum(origRate)
        destProbability = np.array(destRate) / np.sum(destRate)
        origLocation = np.random.choice(range(len(origRate)), 1, p=origProbability)
        destLocation = np.random.choice(range(len(destRate)), 1, p=destProbability)
        origLocation = origLocation[0]
        destLocation = destLocation[0]
        baseFare = bFare
        fare = Dijkstra(G, origLocation, destLocation) * avgFareMultiplier[
            origLocation] * 2 * random.random() + baseFare

        randNum2 = random.random()
        # e^{-driverDensity*thresh} = randNum2
        # -driverDensity*thresh = log(randNum2)
        # thresh = log(randNum2)/(-driverDensity)
        thresh = math.log(randNum2) / (-driverDensity[origLocation])
        newReq = [time, origLocation, destLocation, fare, thresh]
        requestLine = str(time) + ',' + str(origLocation) + ',' + str(destLocation) + ',' + str(fare) + ',' + str(
            thresh) + '\n'
        f.write(requestLine)
    f.close()
import pandas as pd

"""
Computes the probability of your car
capturing a demand that appears
at origVertex, if your car is located
at currVertex.
"""

def probabilityCapture(currLoc,origVertex,G):
    dist = travelTime(currLoc,origVertex,G)
    prob = math.exp(-driverDensity[int(currLoc[0])]*dist)
    return prob
"""
Compute travel time from currLoc to destVertex across graph G
Input currLoc has format (i,j,k).
If (i,j,k)=(5,7,0.3), for example, then
car is 30% complete with edge from 5 to 7.
"""

def travelTime(currLoc, destVertex, G):
    x = currLoc[0]  # current edge start vertex
    y = currLoc[1]  # current edge terminal vertex
    z = currLoc[2]  # proportion of edge completed
    if x == y:
        residTimeCurrEdge = 0
    else:
        residTimeCurrEdge = (1 - z) * G[x][y]  # time left to complete current edge
    otherTime = Dijkstra(G, y, destVertex)  # travel time from terminal vertex of current edge to destination vertex
    totalTime = residTimeCurrEdge + otherTime
    return totalTime


"""This function should be modified and coded by the team.
Based on the current locationa all network level data,
choose a neighboring vertex to relocate to.
"""

# This is calculating the Influence factor, in this function 
# we create an array called valueList, which will basically create weights for the 4 factors. 
def calculateInfluenceFactor(list):
    factor = 1.0
    try:
        valueList = []
        for item in list:
            valueList.append(item[1])
        for i in range(len(valueList) - 1):
            factor = factor + math.fabs(valueList[i] - valueList[i+1])
        factor = factor / (len(valueList) - 1)
    except :
        return 1.0
    return factor


def relocDecision(currLoc, G, origRate, destRate, driverDensity, avgFareMultiplier, baseFare=baseFare):
    # defaultResponse = currLoc[0] #default is to stay in current position.
    # defaultResponse = random.randint(0,numVertices-1)

    # the code that professor had
    optionsForRelocation = list(G[currLoc[0]].keys())
    # defaultResponse = optionsForRelocation[0]
    #ending the code that professor had
    
    # Dictionary to store out values for neighbours
    # these are the factors that we will consider while computing the relocation decision for neighbours
    maxPickUpProbabilty = {}
    dirverDensityAtLocations = {}
    my_AvgFareMultiplier = {}
    my_distance = {}

    # probability To GetRide at current location
    for item in optionsForRelocation:
        maxPickUpProbabilty[item] = probabilityCapture(currLoc, item, G)

    #calculate driver density At location
    for item in optionsForRelocation:
        dirverDensityAtLocations[item] = driverDensity[item]

    # calculate average fare for trip
    for item in optionsForRelocation:
        my_AvgFareMultiplier[item] = avgFareMultiplier[item]

    #get distance of neighbours from current location
    for item in optionsForRelocation:
        my_distance[item] = G[currLoc[0]][item]

    # sort dictionary in ascending order based on value
    sorted_maxPickUpProbabilty = list(sorted(dirverDensityAtLocations.items(), key = lambda kv:(kv[1], kv[0])))
    sorted_dirverDensityAtLocations = list(sorted(dirverDensityAtLocations.items(), key = lambda kv:(kv[1], kv[0])))
    sorted_my_AvgFareMultiplier = list(sorted(my_AvgFareMultiplier.items(), key = lambda kv:(kv[1], kv[0])))
    sorted_my_distance = list(sorted(my_distance.items(), key = lambda kv:(kv[1], kv[0])))
    
    # now as we are going to prioritize pickup probability and average fare multiplier (the higher the better)
    # where as for driver density and distance (the lower the better) as a result they are sorted in ascending order
    # this creates list of tuples sorted based on order that we have prioritised
    sorted_maxPickUpProbabilty.reverse() 
    sorted_my_AvgFareMultiplier.reverse()

    #calculate influence factor
    # influence factor helps offset of some drastic variance in values
    # an example for this is provided in presentation
    # eg (NodeX, 70) (NodeY, 10) over here the second index is average rate
    # so we need to consider for the drastic change from 70 to 10
    influence_factor_for_pickupRate = calculateInfluenceFactor(sorted_maxPickUpProbabilty)
    influence_factor_for_driver_density = calculateInfluenceFactor(sorted_dirverDensityAtLocations)
    influence_factor_for_AvgFareMultiplier = calculateInfluenceFactor(sorted_my_AvgFareMultiplier)
    influence_factor_for_distance = calculateInfluenceFactor(sorted_my_distance)

    score_Value = {}
    for item in optionsForRelocation:
        # initialize all scores with 0
        score_Value[item] = 0
        # here we try to get the index for the particular node in our sorted list and multiply by influence factor 
        #to get a score based on pickUpProbability
        for i in range(len(sorted_maxPickUpProbabilty)):
            if sorted_maxPickUpProbabilty[i][0] == item:
                score_Value[item] = score_Value[item] + (i * influence_factor_for_pickupRate)
                break
        
        # here we try to get the index for the particular node in our sorted list and multiply by influence factor 
        #to get a score based on driverDensity
        for i in range(len(sorted_dirverDensityAtLocations)):
            if sorted_dirverDensityAtLocations[i][0] == item:
                score_Value[item] = score_Value[item] + (i + influence_factor_for_driver_density)
                break
        # here we try to get the index for the particular node in our sorted list and multiply by influence factor 
        #to get a score based on averageFare
        for i in range(len(sorted_my_AvgFareMultiplier)):
            if sorted_my_AvgFareMultiplier[i][0] == item:
                score_Value[item] = score_Value[item] + (i*influence_factor_for_AvgFareMultiplier)
                break
        
        # here we try to get the index for the particular node in our sorted list and multiply by influence factor 
        #to get a score based on distance
        for i in range(len(sorted_my_distance)):
            if sorted_my_distance[i][0] == item:
                score_Value[item] = score_Value[item] + (i*influence_factor_for_distance)
                break

    #now as we have the score for each node in graph subset we need to find the one with smallest score
    # the smaller score indicates the more a particular node occures to the left of the combination in 
    # sorted list as as we prioritised the values on the left the one with the smallest score is actulaay the best
    #for that we sort the score_result dictionary based on value in ascending order 
    score_result = sorted(score_Value.items(), key = lambda kv:(kv[1], kv[0]))

    # the followign line reads the value at 0'th Index and then proceedes to read the Node name in the tuple
    # eg : [(NodeX, 15), (NodeY, 19), (NodeZ, 22)]
    #so the following line returns the NodeName = NodeX
    new_response = score_result[0][0]
    print('Chose to relocate to: ' + str(new_response))
    return new_response


"""This function sh ould be modified and coded by the team.
Based on information about the request, where the car is
located when the request arrives, and other network level
data, make a decision whether to accept the ride or not.
Output/return value should be True or False."""


def rideDecision(nextRequest, locWhenRequest, G, origRate, destRate, driverDensity, avgFareMultiplier,
                 baseFare=baseFare):
    # requested ride offers more than $35, only then accept
    if float(nextRequest['fare']) > 35.0:
        return True
    
    return False


import time

revenueList = []  # keep track of total session revenue for each instance.
for instanceNum in range(numToGenerate):
    print('Starting instance number: ' + str(instanceNum + 1))
    startPosition = 0  # always start at this vertex index
    currTime = 0  # zero seconds have elapsed at the beginning
    currLoc = (startPosition, startPosition, 0.0)  # current location at start
    dest = relocDecision(currLoc, G, origRate, destRate, driverDensity,
                         avgFareMultiplier)  # at very start, a choice to relocate is given
    print(dest)
    nodeArrivalTime = currTime + travelTime(currLoc, dest, G)  # arrival time to relocation position is given

    requests = pd.read_csv("requestInstance" + str(instanceNum + 1) + ".csv", header=None,
                           names=['Time', 'sVert', 'dVert', 'fare', 'thresh'])  # read in the requests instance file
    requestNum = 0  # on the first request (with index 0)
    revenue = 0  # start with no revenue collected
    exitCondition = False
    while (requestNum < len(requests)):  # while our file still contains some requests
        nextRequest = requests.iloc[
            requestNum]  # read in our request.  format of nextRequest is: nextRequest[0] -> time of request, nextRequest[1] -> start vertex, nextRequest[2] -> terminal vertex, nextRequest[3] -> fare revenue, nextRequest[4] -> threshhold distance
        nextRideRequestTime = nextRequest[0]  # time when the next request in the file will occur
        while (
                nextRideRequestTime < currTime):  # if we accepted a ride and currTime (time after accepted ride terminates) is larger than next ride on the list, keep skipping forward through request list until the next request is beyond the currrent time.
            requestNum += 1
            if (requestNum < len(requests)):
                nextRequest = requests.iloc[requestNum]
                nextRideRequestTime = nextRequest[0]
            else:
                exitCondition = True
                break
        accepted = False  # default setting
        if (exitCondition):
            break

        if (
                nextRideRequestTime < nodeArrivalTime):  # if the next request comes in before we arrive at a relocation vertex
            print('Encountered a new request:')
            print(nextRequest)
            if (travelTime(currLoc, dest, G) == 0):
                partialEdge = 1
            else:
                partialEdge = (nextRideRequestTime - currTime) / travelTime(currLoc, dest,
                                                                            G)  # calculate what proportion of currrent edge has been completed
            locWhenRequest = (currLoc[0], dest, partialEdge)  # currLoc[0]
            print('Location when request occurred:')
            print(locWhenRequest)

            if (travelTime(locWhenRequest, nextRequest[1], G) < nextRequest[
                4]):  # if travel time to origin location of request is less than threshhold time
                print('Close enough to choose whether to accept...')
                accepted = rideDecision(nextRequest, locWhenRequest, G, origRate, destRate, driverDensity,
                                        avgFareMultiplier)
                print('Decision to accept or not:')
                print(accepted)
                if (accepted == True):  # if you choose to accept
                    currTime = currTime + travelTime(locWhenRequest, nextRequest[1], G) + travelTime(
                        (nextRequest[1], nextRequest[1], 0.0), nextRequest[2],
                        G)  # fast forward to time after ride has completed
                    currLoc = (nextRequest[2], nextRequest[2], 0.0)  # update position as destination
                    dest = currLoc[0]
                    revenue += nextRequest[3]  # collect revnue from the ride
                    nodeArrivalTime = currTime
                    print('Completed customer delivery to location: ' + str(dest))
                    print('at time: ' + str(nodeArrivalTime))
                    dest = relocDecision(currLoc, G, origRate, destRate, driverDensity, avgFareMultiplier)
                    if (dest == currLoc[
                        0]):  # if the decision to relocate is the current node, no time will elapse, so we are essentially waiting until the next ride request to appear.
                        nodeArrivalTime = currTime + 99999999  # we set this high enough so that nextRideRequestTime < nodeArrivalTime
                    else:
                        nodeArrivalTime = currTime + travelTime(currLoc, dest, G)
            else:  # beyond threshhold distance.  not allowed to accept.
                print('Too far away to accept request...')
                travelTime(locWhenRequest, nextRequest[1], G)
                print('vs.')
                print(nextRequest[4])


        else:  # if we arrive at a relocation vertex before the next request comes in
            print('Arrived at vertex ' + str(dest) + 'as relocation choice.')
            currLoc = (dest, dest, 0.0)  # update current location to reflect arrival
            currTime = nodeArrivalTime  # update current time to the time of arrival
            dest = relocDecision(currLoc, G, origRate, destRate, driverDensity,
                                 avgFareMultiplier)  # make a decision where to relocate next
            if (dest == currLoc[
                0]):  # if the decision to relocate is the current node, no time will elapse, so we are essentially waiting until the next ride request to appear.
                nodeArrivalTime = currTime + 99999999  # we set this high enough so that nextRideRequestTime < nodeArrivalTime
            else:
                nodeArrivalTime = currTime + travelTime(currLoc, dest, G)
        requestNum += 1  # move down the list of upcoming requests

    print('Total revenue collected in instance number ' + str(instanceNum + 1) + ':')
    print(revenue)
    revenueList.append(revenue)
    # time.sleep(5) #pause for 5 seconds after finishing instance

print('Result list across instances:')
print(revenueList)
print('Average revenue collected:')
print(np.mean(revenueList))
