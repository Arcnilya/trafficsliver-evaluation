# Input: instance list from the wang-style file, list with  chosen latencies, and the vector of routes that each packet should travel along
import random
import numpy as np
import sys
#import noise

def getTimefromPacket(packet):return float(packet.split('\t')[0])
def getDirfromPacket(packet):return int(float(packet.split('\t')[1]))
def getSizefromPacket(packet): #In case packet size is also present as third column in file
    if len(packet.split('\t'))==4:
        return int(float(packet.split('\t')[3]))
    if len(packet.split('\t'))==2:
        return int(float(packet.split('\t')[1]))
    if len(packet.split('\t'))==3:
        return int(float(packet.split('\t')[2]))
    else: return 0


def getWeights(n,alphas, seed):
    random.seed(seed) # JM: Added seed
    aph = alphas.split(',')
    if len(aph) != n:
        return -1
    vec_alphas = np.array(aph, dtype=np.float)
    w = np.random.dirichlet(vec_alphas,size=1)[0]
    return w



def buildPacket(size,time,direcction):
    return str(time) + '\t' + str(direcction) + '\t' + str(size)

def joingClientServerRoutes(c,s): #joing the routes choosed by client and server into one route to be used by the simulate funtions
    if len(c)!= len(s):
        sys.exit("ERROR: Client and Server routes must have the same length")
    out = []
    for i in range(0, len(c)): # JM: Changed xrange to range
        if (c[i]==-1):
            out.append(s[i])
        if (s[i]==-1):
            out.append(c[i])
    return out


def simulate_A(instance,mplatencies,routes,seed):
    random.seed(seed) # JM: Added seed
    # Delta time to introduce as the time between two cells are sent from the end side
    delta = 0.0001
    # JM: delta is in seconds, convert to microseconds
    delta *= 10**6
    
    last_packet = 1 
    last_time = 0
    delay = 0
    time_last_incomming = 0
    new_trace = []
    for i in range(len(instance)): #Iterate over each packet # JM: Changed xrange to range
        last_incomming_cell = 0
        packet = instance[i]
        # JM: Changed the methods for getting time and direction to match DT
        original_time = abs(packet)
        direction = 1 if int(packet) > 0 else -1
        size = 512
        # Get the route according to the scheme
        route = routes[i]
        # Get the latency for this route
        chosen_latency = float(random.choice(mplatencies[(route%len(mplatencies))]))
        # JM: latency is in seconds, convert to microseconds
        chosen_latency *= 10**6
        
        if (i != 0 and last_packet != direction): 
            # JM: Added an exception for the first packet
            delay = float(original_time - last_time)/2 
            #Calculate the RTT/2 (latency) request/response,
            #from the time the out cell is sent till the correspongin incell arrives

        new_packet = np.array([])
    
        if (direction == -1):
            new_packet=[original_time - delay + chosen_latency,direction,size,route]
        if (direction == 1 and last_packet == -1): # If is the first out in the burst, it referes to the last icomming time
            new_packet=[time_last_incomming + delta,direction,size,route]
        if (direction == 1 and last_packet == 1): # If we are in an out burst, refers to the last out
            new_packet=[last_time + delta,direction,size,route]
        if i == 0:
            # JM: Added an exception for the first packet
            new_packet=[original_time,direction,size,route]
        
        new_trace.append(new_packet)
        time_last_incomming = original_time - delay + chosen_latency
        last_time = original_time
        last_packet = direction
    np_new_trace =  np.array(new_trace)
    sorted_new_trace = np_new_trace[np_new_trace[:,0].argsort()] 
    #Sorted according to the new timestamps,
    #this is the final instance applied the multi-path effects
    
    # JM: Removed the convertion to relative time since that is done in simulator.py

    return sorted_new_trace


def simulate_B(instance,mplatencies,routes,seed):
    random.seed(seed) # JM: Added seed
    delta = 0.0001 # Delta time to introduce as the time between two cells are sent from the end side
    # JM: delta is in seconds, convert to microseconds
    delta *= 10**6
    
    last_direction = 1 
    last_time = 0
    new_trace = []
    for i in range(len(instance)): #Iterate over each packet # JM: Changed xrange to range
        if int(instance[i]) != 0: # JM: Added a control for zeroes, since NoDef pads with zeroes at the end
            direction = 1 if int(instance[i]) > 0 else -1
            # Get the route according to the scheme
            route = routes[i]
            # Get the latency for this route
            chosen_latency = float(random.choice(mplatencies[(route%len(mplatencies))]))
            # JM: latency is in seconds, convert to microseconds
            chosen_latency *= 10**6

            # JM: If we were sending and just received something new,
            # assume it took chosen_latency time to deliver
            if (direction == -1 and last_direction == 1):
                last_time += chosen_latency
            else:
                last_time += delta

            new_trace.append(np.array([last_time,direction,512,route]))
            last_direction = direction
        
    np_new_trace =  np.array(new_trace)
    sorted_new_trace = np_new_trace[np_new_trace[:,0].argsort()] 
    #Sorted according to the new timestamps,
    #this is the final instance applied the multi-path effects

    return sorted_new_trace