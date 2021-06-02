import numpy as np, numpy.random
import sys
import glob
import random
import time
import os
import pickle
from tqdm import tqdm
import datetime
import multipath
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--circuits", type=int, help="In how many paths is the traffic divided", default=5) 
parser.add_argument("-i", "--inputs", type=str, help="Circuit latencies file", default='circuits_latencies_new.txt') 
parser.add_argument("-o", "--outfolder", type=str, help="Folder for output files", default='outdata') 
parser.add_argument("-r", "--ranges", type=str, help="Range of cells after of which the wr or wrwc schduler design again", default='50,70')
parser.add_argument("-a", "--alpha", type=str, help="alpha values for the Dirichlet function default np.ones(m)", default='1,1,1,1,1')
parser.add_argument("-l", "--loop", type=int, help="number of times to loop the simulator", default=1)
parser.add_argument("-s", "--sim", type=str, help="which simulator version to run [A, B]", default='A')
parser.add_argument("-d", "--ds", type=str, help="which dataset for input")

ds_pkl_files = {"NoDef":["X_NoDef_CW.pkl", "y_NoDef_CW.pkl"],
                "BWR1":["X_BWR1_DT.pkl", "y_BWR1_DT.pkl"],
                "Wang":["X_Wang.pkl", "y_Wang.pkl"]}


def getCircuitLatencies(l, n, seed):
    random.seed(seed)
    file_latencies = open(l,'r')
    row_latencies = file_latencies.read().split('\n')[:-1]
    numberOfClients = int(row_latencies[-1].split(' ')[0])
    randomclient = random.randint(1,numberOfClients)
    ## Get the multiple circuits of the selected client:
    multipath_latencies = []
    for laten in row_latencies:
        clientid = int(laten.split(' ')[0])
        if (clientid == randomclient):
            multipath_latencies.append(laten.split(' ')[2].split(','))	
    ## I only need n circuits, it works when n <  number of circuits in latency file (I had max 6)
    multipath_latencies = multipath_latencies[0:n]
    return multipath_latencies


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")
    
    
def normalize_trace(trace):
    new_trace = []
    zero_time = 0
    for x in range(len(trace)):
        if trace[x] == 0: # skip DT on zeroes
            new_trace.append(0)
        else:
            if zero_time == 0:
                zero_time = trace[x]
            direction = 1 if trace[x] > 0 else -1
            new_time = abs(abs(trace[x]) - abs(zero_time))
            new_trace.append((new_time + 1) * direction)
    return new_trace



def sim_bwr(dataset, outfolder, n, latencies, range_, alphas, iterations, sim):
    
    if sim not in ['A', 'B', 'a', 'b']:
        sys.exit(f"sim: {sim} is not A or B")
        
    if dataset not in ["NoDef", "BWR1", "Wang"]:
        sys.exit(f"invalid dataset: {sim} (NoDef, BWR1, Wang)")
    
    print(f"{now()} simulating BWR multi-path scheme with {n} guards...")
    
    ds = ds_pkl_files[dataset]
    print(f"Loading {ds[0]} and {ds[1]}...")
    traces = np.array(pickle.load(open(ds[0], "rb"), encoding='bytes'))
    labels = np.array(pickle.load(open(ds[1], "rb"), encoding='bytes'))
    
    os.makedirs(outfolder, exist_ok=True)
    sample_tracker = [0] * 100
    
    ranlow = int(range_.split(',')[0])
    ranhigh = int(range_.split(',')[1])

    for seed in range(iterations):
        output_traces = []
        output_labels = []
        # JM: Reusing the iteration id for the random seed
        random.seed(seed)
        print(f"{now()} starting iteration: {seed+1}/{iterations}")
        for x in tqdm(range(len(traces))):
            # JM: Added seed parameter on these 3
            w_out = multipath.getWeights(n, alphas, seed) 
            w_in = multipath.getWeights(n, alphas, seed)
            mplatencies = getCircuitLatencies(latencies, n, seed) # latencies for each of m circuits. length = m
            routes_client = []
            routes_server = []
            sent_incomming = 0
            sent_outgoing = 0

            last_client_route = np.random.choice(np.arange(0,n), p=w_out)
            last_server_route = np.random.choice(np.arange(0,n), p=w_in)
            for i in range(len(traces[x])): # JM: Changed xrange to range
                direction = 1 if int(traces[x][i]) > 0 else -1 # JM: Extract direction

                if (direction == 1):
                    routes_server.append(-1) # Just to know that for this packet the exit does not decide the route
                    sent_outgoing += 1
                    C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                    routes_client.append(last_client_route) 
                    if (sent_outgoing % C == 0): #After C cells are sent, change the circuits
                        last_client_route =  np.random.choice(np.arange(0,n),p = w_out)

                if (direction == -1): 
                    routes_client.append(-1) # Just to know that for this packet the client does not decide the route
                    routes_server.append(last_server_route)
                    sent_incomming += 1
                    C = random.randint(ranlow,ranhigh) #After how many cells the scheduler sets new weights
                    if (sent_incomming % C == 0): #After C cells are sent, change the circuits
                        last_server_route = np.random.choice(np.arange(0,n),p = w_in)


            routes = multipath.joingClientServerRoutes(routes_client,routes_server)
            ##### Routes Created, next to the multipath simulation
            if sim.upper() == 'A':
                new_instance = multipath.simulate_A(traces[x], mplatencies, routes, seed)
            else:
                new_instance = multipath.simulate_B(traces[x], mplatencies, routes, seed)
                
            for guard in range(n):
                temp_trace = []
                for pkt in new_instance:
                    if guard == int(pkt[3]):
                        temp_trace.append(int(pkt[0]) * int(pkt[1])) # directional time
                    else:
                        temp_trace.append(0) # adding gaps
                output_traces.append(normalize_trace(temp_trace))
                output_labels.append(labels[x])
        
        
        iter_dir = os.path.join(outfolder, str(seed))
        print(f"{now()} storing traces in {iter_dir}")
        for i in tqdm(range(len(output_labels))):
            label = output_labels[i]
            os.makedirs(os.path.join(iter_dir, str(label)), exist_ok=True)
            with open(os.path.join(iter_dir, str(label), f"{label}-{sample_tracker[label]}.cell"), "w") as f:
                for pkt in output_traces[i]:
                    f.write(str(pkt) + "\n")
            sample_tracker[label] += 1
            
            
if __name__ == '__main__':
    args = parser.parse_args()
    outfolder_ = args.outfolder
    paths_ = args.circuits
    latencies_ = args.inputs
    outfolder_ = args.outfolder
    range_ = args.ranges
    alpha_ = args.alpha
    iter_ = args.loop
    sim_ = args.sim
    dataset_ = args.ds
    
    seed_value = 177013
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    starttime = time.time()
    sim_bwr(dataset_, outfolder_, paths_, latencies_, range_, alpha_, iter_, sim_)
    endtime = time.time()
    print("Multi-path Simulation done!!! I took (s):", (endtime - starttime))
    