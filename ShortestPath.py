import heapq
'''
UNLESS OTHERWISE STATED, ASSUME ALL TIME/SPACE COMPLEXITIES MENTIONED ARE WORST CASE.
'''

'''
Author: Harshath Muruganantham
'''
def optimalRoute(start:int, end:int, passengers: list, roads:list) -> list:
    '''
    Function description: 
    This function returns an output array consisting of the locations needed to be visited
    in order to minimise the time it takes to reach an end destination from a start destination 
    using a modified version of dijkstra's algorithm.

    Approach description:
    The given roads details are first put into an adjacency list representation of a graph.
    This adjacency list represenation contains for each location (vertex), the roads leading from this
    location to another location, and if the destination location has any passengers in it.
    A modified version of dijkstra's shortest pathh algoprithhm is then implemented (see algorithm for more details),
    where the shortest path from the start location to every location is noted using two different arrays, 
    predecessor-pred, distance-dist (for normal lanes) and predpass, distpass (for a car with passengers using the carpool lane).
    When starting from the start location, the normal lanes are used and pred/dist lists are updated with their respective values.
    When a location is reached, where a passenger is present, the passenger is picked up and dijkstra is continued. 
    Here when relaxing edges (as part of dijkstra's) from a passener-present location, then firstly, the carpool lane is visited, 
    and the arrays predpass and distpass are updated instead. Secondly, the normal lane is vistied as well, and the arrays, pred and dist 
    are updated following the rules of normal dijkstra's (if it is the shortest distance (time taken) to reach the destination location from start location). 
    Predpass will conatin one value pointing to an element in pred. This element indicates the location where the passenger was picked up from. 
    The other values in predpass point to values present within the predpass array. All elements in pred also point to values present within the pred array.

    The backtracking alogorithm of dijkstra's is implemented in this function. Firstly, a check is conducted to check if the end 
    location can be reached quicker through picking up a passenger (using carpool lanes or not) through using the final elements of dist and distpass. 
    If end location is reached faster through picking up a passenger, them backtracking starts from the end location of predpass, ans the locations 
    are added onto final output array and when an element pointing to pred array value in predpass list is encountered, then we start iterating through
    the pred array instead. this process happens until we reach the start location
    If the end location can be reached faster without picking up a passenger, then backtracking should (is) only perfomed on the pred array to 
    get the destinations visited.

    L is the number of locations in the given details (vertexes).
    R is the number of roads in the given details (edges).

    The worst case time complexity is O(R) + O(Llog(L)) + O(Rlog(L)) + O(Rlog(L)) + O(4L) = O(Rlog(L))
    The worst case auxiliary space complexity is O(L) + O(R) + O(4L) + O(L) = O(L+R)

    Inputs:
        start: integer value depicting start location of trip
        end: integer value depicting end location of trip
        passengers: array based list depicting the locations which have passengers waiting to board car.
        roads: array based list withh tuples containing information about the map in the form -> [(start,end,normal_time,carpool_time()]
    Output:
        finalOutput: array based list containing all the location values the car needs to visit to minimise overall time of trip.

    '''

    #Finding max number of vertexes (locations) for graph construction. 
    min = float("inf")
    max = 0
    for details in roads: #O(R) time complexity. 
        if details[0] <= min:
            min = details[0]
        if details[1] > max:
            max = details[1]
        if details[0] > max:
            max = details[0]
        if details[1] <= min:
            min = details[1]

    #Sorting passengers list in order to use binary search when adding edge (road) details.
    passengers.sort() #O(Llog(L)) time complexity.

    #Creation of an adjacency list based graph.
    graph = DirectedGraph(max, max, passengers) #O(R+L) space complexity
    #Filling out graph network with road and location details provided. O(Rlog(L)) time complexity in total.
    for details in roads: # O(R) time complexity for the loop.
        graph.addDirectedWeightedEdge(details) #O(log(L)) time complexity (see function below).

    #O(Rlog(L)) time complexity, O(L) space complexity. Calling Modified dijkstra's shorted path algorithm.
    dist, pred, distPass, predPass = graph.dijkstra(start) 

    #Backtracking method to find the optimal route.
    start = (start, True) #True value depicts if normal lanes were used from this location.
    finalPath=[end] 
    #If end location can be reached faster by picking up a passenger:
    if distPass[end] < dist[end]:
        finalPathTimes = [distPass[end]]
        current = (end, False)
        while current[0] != start[0] or current[1] != start[1]: #Worst Case: O(L) time complexity 
            current = current[0]
            parent = predPass[current] #relying on predpass array (carpool lane times)
            if parent[1] == False: #This loop continues as long as we have a passenger in the car.
                parentTravelTime = distPass[parent[0]]
                finalPath.append(parent[0])
                finalPathTimes.append(parentTravelTime)
                current = parent
            else: #When we no longer have a passenger in a car.
                parentTravelTime = dist[parent[0]]
                finalPath.append(parent[0])
                finalPathTimes.append(parentTravelTime)
                current = parent
                break
        while current[0] != start[0] or current[1] != start[1]: #Worst Case: O(L) time complexity 
            current = current[0]
            parent = pred[current] #now relying on pred array (normal lane times)
            if parent[1] == False: # after executing above loop, incase we need to go back to with-passenger ride.
                parentTravelTime = distPass[parent[0]]
                finalPath.append(parent[0])
                finalPathTimes.append(parentTravelTime)
                current = parent
            else: #keep executing this until we reach start location.
                parentTravelTime = dist[parent[0]]
                finalPath.append(parent[0])
                finalPathTimes.append(parentTravelTime)
                current = parent
    else: #when a passenger need not be picked up at all. Very similar implementation to second loop in above statement.
        finalPathTimes = [dist[end]]
        current = (end, True)
        while current[0] != start[0] or current[1] != start[1]: #Worst Case: O(L) time complexity 
            current = current[0]
            parent = pred[current]
            if parent[1] == True: #Additional check
                parentTravelTime = dist[parent[0]]
                finalPath.append(parent[0])
                finalPathTimes.append(parentTravelTime)
                current = parent
            else: #should never be executed. Exists for safety measures.
                parentTravelTime = distPass[parent[0]]
                finalPath.append(parent[0])
                finalPathTimes.append(parentTravelTime)
                current = parent
                break
                
    #reverse both lists. Only return finalPath, not finalPathTimes.            
    finalPathTimes.reverse() #O(L) time complexity 
    finalPath.reverse() #O(L) time complexity 

    #finalPath contains all the location values the car needs to visit to minimise overall time of trip.
    return finalPath

class DirectedGraph:
    '''
    This class implements the graph we need to solve question one. It also has a heaviliy modified version of dijkstra's shortest path problem that is used to
    solve question one.
    '''
    def __init__(self, vertices: int, maxV: int, passengers: list) -> None:
        '''
        Constructor for initialing an adjacency list based representation of a graph.

        Worst Case Time Complexity: O(1)
        Worst Case (Auxiliary) Space Complexity: O(L) where L is the number of locations in the map.

        Inputs:
            vertices: total number of locations in map
            maxV: the maximum location number in map
            passengers: a sorted array based list depicting the locations which have passengers waiting to board car.
        '''
        self.graph = [None] * (vertices + 1) #Initiating the graph array.
        self.max = maxV
        self.passengers = passengers

    def addDirectedWeightedEdge(self,details: tuple) -> None:
        '''
        This fucntion is used to add road deatils to a specific location in the map.
        In the map, all the roads that leave from a specific location are represented in terms of list containing
        the details i.e.
        [if destination has a passenger waiting, destination, time taken using normal lane, time taken using carpool lane].

        Worst Case Time Complexity: O(log(L)) where L is the number of locations in the map.
        Worst Case (Auxiliary) Space Complexity: O(L + R) where L is the number of locations in the map and R is the number of roads in the map.
                                                 This acts as if self.graph is mainly made here.

        Input:
            details: a tuple containing information about the road. 
                     i.e. (start location, destination, time taken using normal lane, time taken using carpool lane) 

        '''
        #Exectued with the location is new and there are no roads leading from it.
        if self.graph[details[0]] is None:
            self.graph[details[0]] = []

        if len(details) == 4: #When a carpool lane is present in the details given
            #Worst Case Time Complexity: O(log(L)) complexity for binary search.
            if DirectedGraph.binary_search(self.passengers, details[1]) != -1: #if passenger is present in destination
                self.graph[details[0]].append(EdgeData(True, details[1],details[2], details[3])) #O(1)
            else: #if passenger is not present in destination
                self.graph[details[0]].append(EdgeData(False, details[1],details[2], details[3])) #O(1)
        else: #When a carpool lane is not present in the details given
            #Worst Case Time Complexity: O(log(L)) complexity for binary search.
            if DirectedGraph.binary_search(self.passengers, details[1]) != -1: #if passenger is present in destination
                self.graph[details[0]].append(EdgeData(True, details[1],details[2])) #O(1)
            else: #if passenger is not present in destination
                self.graph[details[0]].append(EdgeData(False, details[1],details[2], details[3])) #O(1)
    
    def dijkstra(self, source: int) -> tuple:
        '''
        Function description: 
        This function is a modified version of dijkstra's shortest path algorithm. 

        Approach description:
        A modified version of dijkstra's shortest pathh algorithm is implemented here.
        The shortest path from the start location to every location is noted using two different arrays, pred (predecessor), 
        dist (distance) (for normal lanes) and predpass, distpass (for a car with passengers using the carpool lane).
        There also exists a min-heap priority queue which contains which location(node) to visit next. The quue contains element in the form of
        (key, location details, True/False). The True/False referes to if using normal lane (True) or if there is a passenger on board (False).
        When starting from the start location, the normal lanes are used and pred/dist lists are updated with their respective values.
        When a location is reached, where a passenger is present, the passenger is picked up and dijkstra is continued. 
        Here, when relaxing edges (as part of dijkstra's) from a passenger-present location, the carpool lane is visited, 
        and the arrays predpass and distpass are updated. However, the normal lane is vistied as well, and the arrays, pred and dist 
        are updated following the rules of normal dijkstra's (if it is the shortest distance (time taken) to reach the destination location from start location). 
        Predpass will conatin one value pointing to an element in pred. This element indicates the location where the passenger was picked up from. 
        The other values in predpass point to values present within the predpass array. All elements in pred also point to values present within the pred array.

        L is the number of locations in the given details (vertexes).
        R is the number of roads in the given details (edges).

        Worst Case Time Complexity: O(4L) + O(log(L)) + O(Llog(L)) + O(2*Rlog(L)) = O(Rlog(L))
                                    = O(Rlog(L))
        Worst Case (Auxiliary) Space Complexity: O(5L) = O(L)

        Inputs:
            source: integer representing starting location for dijkstra's 
        Outputs:
            dist: array based list containing the distances to every location from source location WHEN USING NORMAL LANE
            pred: array based list containing the predecessor of every location WHEN USING NORMAL LANE
            distPass: array based list containing the distances to 
                        every location from the location where the passenger is FIRST picked up WHEN USING CARPOOL LANE
            predPass: array based list containing the predecessor of every location WHEN USING CARPOOL LANE
                        (One element in here points to an element in pred - this is the location the passenger was picked up.)

        '''
        #Creating arrays needed for keeping track.
        dist = [float("inf") for i in range(self.max + 1)] #O(L) space/time complexity
        distPass = [float("inf") for i in range(self.max + 1)] #O(L) space/time complexity
        pred = [0] * (self.max + 1) #O(L) space complexity
        predPass = [None] * (self.max + 1) #O(L) space complexity
        #Creating a priority Queue. 
        priorityQueue = []
        #Pushing source element onto priority queue. Passenger cannot be present in first location, so we will be using normal lane and True is set.
        heapq.heappush(priorityQueue, (0, source, True)) #O(log(L)) time complexity
        #Setting up standard values for first element.
        pred[source] = (0,True)
        dist[source] = 0

        while len(priorityQueue) != 0:
            u = heapq.heappop(priorityQueue) #O(log(L)) time complexity
            key = u[0] #equal to distance 
            vertex = u[1] #location details
            ifNormal = u[2] #if using normal lanes (denote false if no passenger at this location or picked up before this location on this route)

            if dist[vertex] == key:
                if self.graph[vertex] is None:
                    continue
                #In total, all edges (Roads) are visited atleast once / max twice. 
                #Worst Case Complexity: O(2R) = O(R). Total worst case time complexity: O(Rlog(L))
                for i in self.graph[vertex]: 
                    v = i.getDestination() #O(1) time complexity
                    if ifNormal: #No passengers at this location and no passengers picked up before this location on this route.
                        weight = i.getNormalTime() #O(1) time complexity
                        #Update distance and predecessor arrays if new distance is less than existing one.
                        if dist[v] > dist[vertex] + weight:
                            dist[v] = dist[vertex] + weight #O(1) time complexity
                            pred[v] = (vertex, True) #O(1) time complexity
                            if i.hasPassenger():
                                #Heap is pushed with ifNormal set to false, as there is a passenger present at this location.
                                heapq.heappush(priorityQueue,(dist[v],v, False)) #O(log(L)) time complexity
                            else:
                                #Heap is pushed with ifNormal set to True, as there is nopassenger present at this location
                                #and we will be using normal lanes.
                                heapq.heappush(priorityQueue,(dist[v],v, True)) #O(log(L)) time complexity
                            
                    else: #Passenger present on location or in car already.
                        #Caluclate time if this exact route was taken without picking up passenger.
                        weight = i.getNormalTime() #O(1) time complexity
                        if dist[v] > dist[vertex] + weight:
                            dist[v] = dist[vertex] + weight #O(1) time complexity
                            pred[v] = (vertex, True) #O(1) time complexity
                            heapq.heappush(priorityQueue,(dist[v],v, False)) #O(log(L)) time complexity
                        weight = i.getCarpoolTime()
                        if weight == None:
                            continue
                        
                        #Here we caluclate the time when using carpool lanes

                        #Only enter this statement when 'first' entering the passenger list from the normal list.
                        # i.e. after picking up a passenger for the first time. 
                        if distPass[vertex] == float("inf"):
                            #Standard values for first element are set in order to enter if statement below for the first time.
                            distPass[vertex] = 0 #O(1) time complexity.  
                            dist_temp = dist[vertex] #O(1) time complexity
                        if distPass[v] > distPass[vertex] + weight:
                            distPass[v] = distPass[vertex] + weight + dist_temp #O(1) time complexity
                            if distPass[vertex] == 0: #This statement is only entered the FIRST time a passenger is picked up.
                                #Reverting what was done in line 256 if statement.
                                distPass[vertex] = float("inf")
                                #Predesessor here is set to true as we have come from a location without a passenger
                                #This value in predpass points to the pred array and will be used when backtracking.
                                predPass[v] = (vertex, True) #O(1) time complexity
                            else:
                                predPass[v] = (vertex, False) #O(1) time complexity
                            dist_temp = 0
                            #same pushing as above. IfNormal will always be set to false from here forth as we 
                            #have already picked up a passenger.
                            heapq.heappush(priorityQueue,(dist[v],v, False)) #O(log(L)) time complexity

        return dist, pred, distPass, predPass
    
    @staticmethod
    def binary_search(arr: list, target: int) -> int:
        '''
        Standard Binary search algorithm. Finds the index of target element in array.
        If target element is not in array, -1 is returned. 

        Worst Case Time Complexity: O(log(V)) where V is the number of elements in arr.
        Worst Case (Auxiliary) Space Complexity: O(1)

        Inputs:
            arr: array based list to search for the target element in.
            traget: an integer representing the element we are trying to index in arr.
        Outputs:
            index of target in array (integer)
        '''
        low = 0
        high = len(arr) - 1

        #Size of serahing array is halved at each iteration of this loop.
        while low <= high: #O(log(V)) time complexity where V is the number of elements in arr.
            mid = (low + high) // 2

            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1

        return -1

class EdgeData:
    '''
    This class contains the data of the roads leading from any location including the destination,
    the time taken to reach if using a normal lane and the time taken to reach if using a carpool lane. 
    '''
    def __init__(self, passenger: bool, destination: int, normalTime: int, carpoolTime = None) -> None:
        '''
        Constructor class of Edge Data, a class containing information about the roads leading away from a location.

        Worst Case Time Complexity: O(1).
        Worst Case (Auxiliary) Space Complexity: O(1).

        INPUTS:
            passenger: True if destination location contains a passenger. False if not.
            destination: integer representing destination location of road.
            normalTime: integer repressenting time it would take to reach destination location if normal lane is taken.
            carpoolTime: integer repressenting time it would take to reach destination location if carpool lane is taken.

        '''
        self.destination = destination
        self.normalTime = normalTime
        if carpoolTime != None:
            self.carpoolTime = carpoolTime
        else:
            self.carpoolTime = None
        self.passenger = passenger
    
    def getDestination(self) -> int:
        '''
        Get the destination of the current road.

        Worst Case Time Complexity: O(1).
        Worst Case (Auxiliary) Space Complexity: O(1).

        OUTPUTS:
            destination: integer representing destination location of road.
        '''
        return self.destination
    
    def getNormalTime(self) -> int:
        '''
        Get the time it would take to reach destination location if normal lane is taken.

        Worst Case Time Complexity: O(1).
        Worst Case (Auxiliary) Space Complexity: O(1).
        
        OUTPUTS:
            normalTime: integer repressenting time it would take to reach destination location if normal lane is taken.
        '''
        return self.normalTime
    
    def getCarpoolTime(self) -> int:
        '''
        Get the time it would take to reach destination location if carpool lane is taken.

        Worst Case Time Complexity: O(1).
        Worst Case (Auxiliary) Space Complexity: O(1).
        
        OUTPUTS:
            carpoolTime: integer repressenting time it would take to reach destination location if carpool lane is taken.
        '''
        return self.carpoolTime
    
    def hasPassenger(self) -> bool:
        '''
        Find out if there is a passenger in the destination location.

        Worst Case Time Complexity: O(1).
        Worst Case (Auxiliary) Space Complexity: O(1).
        
        OUTPUTS:
            passenger: True if destination location contains a passenger. False if not.
        '''
        return self.passenger
