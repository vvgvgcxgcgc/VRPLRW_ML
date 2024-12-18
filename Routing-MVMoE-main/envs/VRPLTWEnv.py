from dataclasses import dataclass
import torch
import os, pickle
import numpy as np
import random
import json
torch.autograd.set_detect_anomaly(True)

__all__ = ['VRPLTWEnv']


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_service_time: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_start: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_end: torch.Tensor = None
    # shape: (batch, problem)
    prob_emb: torch.Tensor = None
    # shape: (num_training_prob)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    PROBLEM: str = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    open: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)
    current_routes: torch.Tensor = None
    # shape:(batch,pomo,problem)
    cur_demands: torch.Tensor = None
    # shape:(batch,problem)
    cur_travel_time_routes: torch.Tensor = None
    # shape: (batch, pomo, problem)
    tw_start_routes: torch.Tensor = None
    tw_end_routes: torch.Tensor = None
    truck_num: torch.Tensor = None
    total_time: torch.Tensor = None
    cur_index: torch.Tensor = None
    mask: torch.Tensor = None
    init_mask: torch.Tensor = None
    done = False




    max_route: int = 0

def update_duration_mask(cur_mask, cur_routes, routes_cost, service_time,   max_duration, truck_num, selected_route, node_xy, depot, speed = 1.0):
    batch, pomo, n, _ = cur_routes.shape
    device = cur_mask.device
    BATCH_IDX = torch.arange(batch)[:, None, None].expand(batch, pomo, n).to(device)
    POMO_IDX = torch.arange(pomo)[None, :, None].expand(batch, pomo, n).to(device)
    ROUTE_IDX =  torch.arange(n)[None, None, :].expand(batch, pomo, n).to(device)

    selected_route_cost = torch.gather(routes_cost, 2, selected_route.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n + 1)).squeeze(2)
    nodes_route = torch.gather(cur_routes, 2, selected_route.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n)).squeeze(2)
    selected_route_service_time = torch.gather(service_time.unsqueeze(1).expand(-1, pomo, -1), 2, nodes_route) # shape (batch, pomo, n)

    nodes_num = torch.gather(truck_num, 2, selected_route.unsqueeze(-1)).expand(batch, pomo, n)
    total_cost = torch.sum(selected_route_cost, dim = 2)
    # suspect_pos = torch.cat((nodes_route[BATCH_IDX, POMO_IDX,cur_index].unsqueeze(-1), nodes_route[BATCH_IDX, POMO_IDX,cur_index -1].unsqueeze(-1), nodes_route[BATCH_IDX, POMO_IDX,cur_index -1].unsqueeze(-1)), 2)
    node_xy_expand = node_xy.unsqueeze(1).expand(-1, pomo, -1, -1)
    nodes_route_coords = torch.gather(node_xy_expand, 2, nodes_route.unsqueeze(-1).expand(-1, -1, -1, 2)) #shape (batch, pomo, 3, 2)
    nodes_num = nodes_num - 1
    # nodes_route_coords[:, :, 2, :] = torch.where(cur_index.unsqueeze(-1).expand(-1, -1, 2) < nodes_num.unsqueeze(-1).expand(-1, -1, 2), nodes_route_coords[:, :, 2, :], depot.unsqueeze(1).expand(-1, pomo, -1))
    distances= torch.sqrt(torch.sum((nodes_route_coords.unsqueeze(2) - node_xy_expand.unsqueeze(3)) ** 2, dim=-1))/speed  # shape (batch, pomo, n, n)
    xy_depot = torch.sqrt(torch.sum((depot.unsqueeze(1).unsqueeze(2).unsqueeze(2).expand(-1, pomo, 1,1, -1) - node_xy_expand.unsqueeze(3)) ** 2, dim = -1))/ speed #shape(batch, pomo, n, 1)
    range_tensor = torch.arange(n).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)# Shape: (1, 1, 1, n)

    padding = range_tensor > nodes_num.unsqueeze(-1)
    padding = padding*10 #shape (batch, pomo, n, n)
    distances = distances + padding
    new_distances = distances.clone()
    new_distances = torch.cat((xy_depot, new_distances), dim = 3) # shape (batch, pomo, n, n+1)
    distances[BATCH_IDX, POMO_IDX,ROUTE_IDX, nodes_num + 1] = xy_depot.squeeze(-1)
    selected_route_service_time = selected_route_service_time.unsqueeze(2).expand(-1, -1, n, -1)
    zero = torch.zeros((batch, pomo, n)).to(device)

    selected_route_service_time[BATCH_IDX, POMO_IDX,ROUTE_IDX, nodes_num + 1] = zero
    xy_depot = xy_depot*500

    selected_route_service_time = torch.cat((selected_route_service_time,xy_depot ), dim = 3) # shape (batch, pomo, n, n + 1)

    distances = torch.cat((distances,xy_depot ), dim = 3) # shape (batch, pomo, n + 1)
    cost = distances + new_distances - selected_route_cost.unsqueeze(2) + selected_route_service_time + service_time.unsqueeze(1).unsqueeze(-1).expand(batch, pomo, n, 1) + total_cost.unsqueeze(-1).unsqueeze(-1) <= max_duration.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    mask = torch.sum(cost, dim = -1) == 0 # shape(batch, pomo, n)
    mask = mask.to(torch.int64)


    cur_mask[BATCH_IDX.squeeze(-1), POMO_IDX.squeeze(-1), :,  selected_route] = torch.where(cur_mask[BATCH_IDX.squeeze(-1), POMO_IDX.squeeze(-1), :,  selected_route] == 0, mask, 1)

    indices = (mask == 1).nonzero()
    # print("update duration indices: ", indices)

    return cur_mask



def update_demand_mask(cur_demands, node_demand, cur_mask, max_demand, selected_route):
    '''
        cur_demands Tensor shape (batch, pomo, n)
        node_demand Tensor shape (batch, n)
        cur_mask Tensor shape (batch, pomo, n, n)
        max_demand Tensor shape(batch)
    '''
    batch,pomo,n = cur_demands.shape
    device = cur_demands.device
    BATCH_IDX = torch.arange(batch)[:, None].expand(batch, pomo).to(device)
    POMO_IDX = torch.arange(pomo)[None, :].expand(batch, pomo).to(device)
    selected_route_demand = torch.gather(cur_demands, 2, selected_route.unsqueeze(-1))
    node_demand = node_demand.unsqueeze(1).expand(-1, pomo, n)
    round_error_epsilon = 0.00001
    max_demand = max_demand.unsqueeze(-1).unsqueeze(-1)
    demand_mask = selected_route_demand + node_demand   > max_demand #shape(batch, pomo, n)
    cur_mask[BATCH_IDX, POMO_IDX, :,  selected_route] = torch.where(cur_mask[BATCH_IDX, POMO_IDX, :,  selected_route] == 0, demand_mask.to(torch.int64), 1)

    indices = (demand_mask == 1).nonzero()
    # print("demand mask: ", indices)
    
    return cur_mask

    
def update_route(node_xy, depot,node_demand,service_time, cur_demands, truck_num, cur_routes,routes_cost,  selected_node,node_mask, speed = 1.0):
    '''
    Update routes each decode time
    node_xy: Tensor shape (batch, n, 2) contains coord of n customer
    depot: Tensor shape (batch, 2)     contains coord of depot point
    truck_num: Tensor shape (batch, pomo, n) contains  number of customer in route, maximize n route, if just use m < n route, padding 0
    cur_routes: Tensor shape (batch, pomo, n, n) each routes contains index of customer, padding 0
    routes_cost: Tensor shape (batch, pomo, n, n + 1) cost of from right previous of each customer và cộng thêm servicetime, add value from last node to depot, padding 0
    selected_node: Tensor shape(batch, pomo) contains customer index chosen by model
    # selected_route: Tensor shape(batch, pomo) contain route index chosen by model
    Return:
        new_routes // same as cur_routes, but is inserted selected customer into selected route
        shape(batch, pomo, n, n)
        new_routes_cost shape(batch, pomo, n, n + 1)
        truck_num
        cur_demands
        insert_index
        selected_route_index
    '''
    device = node_xy.device
    batch, pomo, _, n = cur_routes.shape
    BATCH_IDX = torch.arange(batch)[:, None, None].expand(batch, pomo, n).to(device)
    POMO_IDX = torch.arange(pomo)[None, :, None].expand(batch, pomo, n).to(device)
    ROUTE_IDX =  torch.arange(n)[None, None, :].expand(batch, pomo, n).to(device)
    # raw_selected_route = selected_route.clone()
    # selected_route = selected_route.unsqueeze(-1)
    # node_num = torch.gather(truck_num, 2, selected_route) # shape (batch, pomo, 1)
    # node_num = node_num -1
    node_num = torch.where(truck_num - 1 < 0, 0, truck_num - 1)
    # selected_route_expand = selected_route.unsqueeze(-1).expand(-1, -1, 1, n)
    # nodes_route = torch.gather(cur_routes, 2, selected_route_expand).squeeze(2)
    nodes_route = cur_routes.clone() # shape (batch, pomo, n, n)

    node_xy_expand = node_xy.unsqueeze(1).expand(-1, pomo, -1, -1)
    raw_selected_node = selected_node.clone()
    selected_node = selected_node.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    selected_node_coord = torch.gather(node_xy_expand, 2, selected_node).unsqueeze(2).expand(-1, -1, n, -1, -1) # shape( batch, pomo, 1, 2)
    nodes_route_expand = nodes_route.unsqueeze(-1).expand(-1,-1,-1, -1, 2) #shape (batch, pomo, n, n, 2 )
    nodes_route_coords = torch.gather(node_xy_expand.unsqueeze(2).expand(-1, -1, n, -1, -1), 3, nodes_route_expand) #shape (batch, pomo, n, n, 2)
    node_num_expand = node_num.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1,1, 2) # shape(batch, pomo, 1, 2)
    last_node_coord = torch.gather(nodes_route_coords, 3, node_num_expand) #shape(batch, pomo,n,1, 2)
    depot_expand = depot.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, pomo, n, 1, -1) #shape(batch, pomo,n,1, 2)
    distance_lastnode_depot = torch.sqrt(torch.sum((last_node_coord - depot_expand) ** 2, dim=-1))/speed
    #shape(batch, pomo,n, 1)
    distance_selected_depot = torch.sqrt(torch.sum((selected_node_coord - depot_expand) ** 2, dim=-1))/speed
    #shape(batch, pomo,n, 1)
    distances = torch.sqrt(torch.sum((selected_node_coord - nodes_route_coords) ** 2, dim=-1)) / speed
    # shape(batch, pomo,n,  n)

    range_tensor = torch.arange(n).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device) # Shape: (1, 1, 1, n)
    
    padding = range_tensor > node_num.unsqueeze(-1)
    padding = padding*10 #shape (batch, pomo, n, n)
    distances = distances + padding

    new_distances = distances.clone()
    new_distances = torch.cat((distance_selected_depot, new_distances), dim = 3) # shape (batch, pomo, n, n+1)
    node_num = node_num + 1
    distances[BATCH_IDX, POMO_IDX,ROUTE_IDX, node_num] = distance_lastnode_depot.squeeze(-1)
    distance_lastnode_depot = distance_lastnode_depot*10
    distances = torch.cat((distances,distance_lastnode_depot ), dim = 3) # shape (batch, pomo, n + 1)
    # lost_index = raw_selected_route.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n + 1)
    # lost_distance = torch.gather(routes_cost, 2,lost_index ).squeeze(2) #shape(batch, pomo, n+1)
    inserted_cost = new_distances + distances - routes_cost #shape (batch, pomo,n, n + 1)
    node_mask = node_mask*100
    inserted_cost = inserted_cost + node_mask.unsqueeze(-1)
    min_indices = torch.argmin(inserted_cost.view(batch,pomo, -1), dim=2)
    selected_route_index = min_indices // inserted_cost.size(-1) # shape(batch, pomo)
    insert_index = min_indices % inserted_cost.size(-1) # shape (batch, pomo)
    # insert_index = torch.argmin(inserted_cost, dim = 3) # shape (batch, pomo, )
    selected_node_route = torch.gather(nodes_route, 2,selected_route_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n)).squeeze(2)
    lost_distance = torch.gather(routes_cost, 2, selected_route_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n + 1)).squeeze(2)
    selected_dis = torch.gather(distances, 2, selected_route_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n + 1)).squeeze(2)
    selected_new_dis = torch.gather(new_distances, 2, selected_route_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n + 1)).squeeze(2)
    new_nodes_route = torch.zeros((batch,pomo, n + 1),  dtype=torch.int64).to(device)
    expanded_indices = torch.arange(n + 1).expand(batch,pomo, -1).to(device)  # Shape: (batch, n + 1)
    
    # print("expanded_indices shape:", expanded_indices.shape)
    # print("insert_index shape:", insert_index.unsqueeze(1).shape)
    
    mask_before = expanded_indices < insert_index.unsqueeze(1)  # Shape: (batch, n+1)
    mask_after = expanded_indices > insert_index.unsqueeze(1)
    new_nodes_route[mask_before] = selected_node_route[mask_before[:, :, :-1]]
    new_nodes_route[mask_after] = selected_node_route[mask_after[:, :, 1:]]
    new_nodes_route[BATCH_IDX, POMO_IDX,  insert_index] = raw_selected_node
    cur_routes[BATCH_IDX, POMO_IDX, selected_route_index] = new_nodes_route[:, :, :-1]
    new_route_cost = torch.zeros((batch, pomo, n + 2)).to(device)
    expanded_indices = torch.arange(n + 2).expand(batch,pomo, -1).to(device)  # Shape: (batch, n + 2)
    mask_before = expanded_indices < insert_index.unsqueeze(1)  # Shape: (batch, n+2)
    mask_after = expanded_indices > insert_index.unsqueeze(1)
    new_route_cost[mask_before] = lost_distance[mask_before[:, :, :-1]]
    lost_distance[BATCH_IDX, POMO_IDX, insert_index] = selected_dis[BATCH_IDX, POMO_IDX, insert_index] 
    new_route_cost[mask_after] = lost_distance[mask_after[:, :, 1:]]
    new_route_cost[BATCH_IDX, POMO_IDX, insert_index] = selected_new_dis[BATCH_IDX, POMO_IDX, insert_index] +  service_time.unsqueeze(1).expand(-1, pomo, -1)[BATCH_IDX, POMO_IDX, raw_selected_node]
    routes_cost[BATCH_IDX, POMO_IDX, selected_route_index] = new_route_cost[:, :, :-1]
    truck_num[BATCH_IDX, POMO_IDX, selected_route_index]  = truck_num[BATCH_IDX, POMO_IDX, selected_route_index] + 1


    node_demand = node_demand.unsqueeze(1).expand(-1, pomo, -1)
    cur_demands[BATCH_IDX, POMO_IDX, selected_route_index] = cur_demands[BATCH_IDX, POMO_IDX, selected_route_index] + node_demand[BATCH_IDX, POMO_IDX, raw_selected_node]

    return cur_routes, routes_cost, truck_num, cur_demands, insert_index,selected_route_index





def init_travel_time_demand( depot, node_xy, node_demand, service_time,  node_indices, max_duration, truck_num, speed=1.0):
    """
    Computes the travel time from the depot to a set of nodes identified by indices.

    Args:
        depot (torch.Tensor): Tensor of shape (batch, 1, 2) representing depot coordinates.
        node_xy (torch.Tensor): Tensor of shape (batch, n, 2) representing coordinates of all nodes.
        node_demand: Tensor of shape(batch, n)
        node_indices (torch.Tensor): Tensor of shape (batch, m) containing indices of nodes to calculate travel time for.
        truck_num: Tensor shape (batch, n)
        speed (float): Travel speed, default is 1.0.

    Returns:
        torch.Tensor: Tensor of shape (batch, m) representing travel times from depot to each selected node.
    """
    # Gather node coordinates using the indices
    device = depot.device

    depot = depot.unsqueeze(1)

    print(">>>> depot  ", depot.shape)
    print(">>>> node_xy  ", node_xy.shape)
    print(">>>> node_demand  ", node_demand.shape)
    print(">>>> node_indices  ", node_indices.shape)
    print(">>>> truck_num  ", truck_num.shape)
    print(">>>> service_time  ", service_time.shape)
    print(">>>> max_duration  ", max_duration.shape)

    routes_demand = torch.gather(node_demand, 1, node_indices)
    routes_demand = routes_demand * truck_num

    selected_node_xy = torch.gather(node_xy, 1, node_indices.unsqueeze(-1).expand(-1, -1, 2))  # Shape: (batch, m, 2)
    selected_service_time= torch.gather(service_time, 1, node_indices )
    # Compute the difference in coordinates
    delta = selected_node_xy - depot  # Shape: (batch, m, 2)

    # Compute Euclidean distances
    distances = torch.sqrt(torch.sum(delta ** 2, dim=-1))  # Shape: (batch, m)

    # Compute travel time by dividing distance by speed
    travel_times = distances / speed 
    batch, m = travel_times.shape

    # Apply mask to travel_time
    travel_times = travel_times * truck_num  # Masked values are set to 0
    
    n = node_xy.size(1)

    # Initialize result matrix with zeros
    travel_matrix = torch.zeros((batch, n, n + 1), device=device)

    # Fill the first m values of each row in the n x n matrix
    travel_matrix[:, :m, 0] = travel_times + selected_service_time
    travel_matrix[:, :m, 1] = travel_times


    distances= torch.sqrt(torch.sum((selected_node_xy.unsqueeze(1) - node_xy.unsqueeze(2)) ** 2, dim=-1)).unsqueeze(-1)/speed  # shape (batch, n, n, 1)
    xy_depot = torch.sqrt(torch.sum((depot.unsqueeze(2) - node_xy.unsqueeze(2)) ** 2, dim = -1)).unsqueeze(2).expand(-1, -1, n, 1) / speed #shape(batch, n,n, 1)

    distances = distances + xy_depot  + service_time.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n, 1) + selected_service_time.unsqueeze(1).unsqueeze(-1).expand(-1, n, n, 1)

    cost =  distances > max_duration.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    routes_matrix = torch.zeros((batch, n, n), device=device, dtype=torch.int64)
    routes_matrix[:, :m, 0] = node_indices

   
    return travel_matrix, routes_matrix, routes_demand, cost.squeeze(-1)
def calculate_tw_demand_index(tw_start, tw_end,demands,  node_indices, truck_num,  pomo = 1):
    '''
    Calculate the time windows for current routes
    tw_start: Tensor shape (batch, n)
    tw_end: Tensor shape (batch, n)
    demands: Tensor shape (batch, n)
    node_indices: Tensor shape(batch, m)
    Return:
        tw_start_routes: Tensor shape (batch, pomo, n, n)
        tw_end_routes: Tensor shape (batch, pomo, n, n)
    '''
    # Gather node coordinates using the indices
    selected_tw_start = torch.gather(tw_start, 1, node_indices.unsqueeze(-1).expand(-1, -1, 2))  # Shape: (batch, m, 2)
    selected_tw_end = torch.gather(tw_end, 1, node_indices.unsqueeze(-1).expand(-1, -1, 2))  # Shape: (batch, m, 2)
    selected_routes = node_indices.unsqueeze(-1).expand(-1, -1, 2)
    selected_demands_routes = torch.gather(demands, 1, node_indices)  # Shape: (batch, m)

    batch, m = selected_tw_start.shape
    n = tw_start.size(1)
     # Create a range tensor to compare with truck_num
    range_tensor = torch.arange(m, device=selected_tw_start.device).unsqueeze(0)  # Shape: (1, m)

    # Create a mask based on truck_num
    mask = range_tensor < truck_num.unsqueeze(1)  # Shape: (batch, m)

    selected_tw_start = selected_tw_start * mask

    # Initialize result matrix with zeros
    tw_start_routes = torch.zeros((batch, n, n))

    # Fill the first m values of each row in the n x n matrix
    tw_start_routes[:, :m, 0] = selected_tw_start
    tw_start_routes = tw_start_routes.unsqueeze(1).expand(batch, pomo, n, n)  # Shape: (batch, pomo, n, n)
    
    selected_tw_end = selected_tw_end * mask

    # Initialize result matrix with zeros
    tw_end_routes = torch.zeros((batch, n, n))

    # Fill the first m values of each row in the n x n matrix
    tw_end_routes[:, :m, 0] = selected_tw_end
    tw_end_routes = tw_end_routes.unsqueeze(1).expand(batch, pomo, n, n)  # Shape: (batch, pomo, n, n)

    # Initialize result matrix with zeros
    routes = torch.zeros((batch, n, n))

    # Fill the first m values of each row in the n x n matrix
    routes[:, :m, 0] = selected_routes
    routes = routes.unsqueeze(1).expand(batch, pomo, n, n)  # Shape: (batch, pomo, n, n)
    selected_demands_routes = selected_demands_routes * mask


    # Initialize result matrix with zeros
    demands_routes = torch.zeros((batch, n))

    # Fill the first m values of each row in the n x n matrix
    demands_routes[:, :m] = selected_demands_routes
    demands_routes = tw_end_routes.unsqueeze(1).expand(batch, pomo, n)  # Shape: (batch, pomo, n)



    return routes, demands_routes,  tw_start_routes, tw_end_routes




    







class VRPLTWEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "VRPLTW"
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else env_params['device']

        self.train_pkl_path = "gen_data/new/data50_ntruck_10k.pkl"
        self.val_pkl_path = "gen_data/new/data50_ntruck_2k.pkl"

        self.train_dataset = None
        self.train_ntruck_category = None
        self.val_dataset = None
        self.val_ntruck_category = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.num_problem = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_service_time = None
        # shape: (batch, problem+1)
        self.depot_node_tw_start = None
        # shape: (batch, problem+1)
        self.depot_node_tw_end = None
        # shape: (batch, problem+1)

        self.speed = 1.0
        self.depot_start, self.depot_end = 0., 3.  # tw for depot [0, 3]

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, problems=None, aug_factor=1):
        if problems is not None:

            self.problem_data = problems
            
            depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, init_routes, mask_node, truck_num, max_demand = problems
        else:
            depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, init_routes, mask_node, truck_num, max_demand = self.get_random_problems(batch_size, self.problem_size, normalized=True)
        self.batch_size = depot_xy.size(0)
        
        self.num_problem = node_xy.size(1)
        self.mask_node = mask_node.to(self.device)

        self.node_xy = node_xy.to(self.device)
        self.depot = depot_xy.to(self.device)

        # shape: (batch, problem+1, 2)
        # shape: (batch, 1)
        self.node_demand = node_demand.to(self.device) #shape (batch, n)
        self.max_demand = max_demand.to(self.device)
        # shape: (batch,)
        self.route_limit = route_limit.to(self.device)
        self.init_routes = init_routes.to(self.device) #shape(batch, pomo, n)
        # shape: (batch, 1)
        self.service_time = service_time.to(self.device)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)
        self.truck_num= truck_num.to(self.device)

        self.reset_state.depot_xy = depot_xy.to(self.device)
        self.reset_state.node_xy = node_xy.to(self.device)
        self.reset_state.node_demand = node_demand.to(self.device)
        self.reset_state.node_service_time = service_time.to(self.device)
        self.reset_state.node_tw_start = tw_start.to(self.device)
        self.reset_state.node_tw_end = tw_end.to(self.device)
        self.reset_state.prob_emb = torch.FloatTensor([1, 0, 0, 1, 1]).unsqueeze(0).to(self.device)  # bit vector for [C, O, B, L, TW]

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

        self.step_state.PROBLEM = self.problem
        self.step_state.truck_num = self.truck_num.unsqueeze(1).expand(self.batch_size, self.pomo_size,-1).to(self.device)

        self.step_state.cur_travel_time_routes, self.step_state.current_routes,  self.step_state.cur_demands, self.duration_mask = init_travel_time_demand(self.depot, self.node_xy, self.node_demand, self.service_time, self.init_routes, self.route_limit, self.truck_num, self.speed)
        self.step_state.cur_travel_time_routes = self.step_state.cur_travel_time_routes.unsqueeze(1).expand(-1, self.pomo_size, -1, -1)
        self.step_state.current_routes = self.step_state.current_routes.unsqueeze(1).expand(-1, self.pomo_size, -1, -1)
        self.step_state.cur_demands = self.step_state.cur_demands.unsqueeze(1).expand(-1, self.pomo_size, -1)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.current_route = None
        #shape: (batch, pomo)

        # shape: (batch, pomo, problem+1)
        self.init_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size,self.problem_size ), dtype = torch.int64).to(self.device)
        # print("init_routes: ", self.init_routes[0])
        zero_mask = (self.step_state.truck_num == 0)  # Shape: (batch, pomo, m)
        
        zero_mask = zero_mask.unsqueeze(2).expand(self.batch_size, self.pomo_size, self.problem_size, -1)
        # print("zero_mask: ", zero_mask.sum(dim = -2)[0])

        self.init_mask = self.init_mask | zero_mask
        # print("self.init_mask: ", torch.sum(self.init_mask, dim=-1)[0])
        self.mask_node = self.mask_node.unsqueeze(1).expand(-1, self.pomo_size, -1)
        mask_node = self.mask_node.unsqueeze(-1).expand(-1, self.pomo_size, self.problem_size, self.problem_size)
        # print("mask_node: ", mask_node.sum(dim = -1)[0])
        self.init_mask = self.init_mask | mask_node
        # print("init_mask: ", self.init_mask.sum(dim = -1)[0])

        self.ninf_mask = self.init_mask.clone()
        # print("duration_mask: ", self.duration_mask[0])
        # print("self.ninf_mask: ", torch.sum(self.ninf_mask, dim=-1)[0])

        self.ninf_mask = torch.where(self.ninf_mask == 0, self.duration_mask.unsqueeze(1).expand(self.batch_size, self.pomo_size, -1, -1).to(torch.int64), 1)

        
        # # shape: (batch, pomo, problem+1)
        # self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # # shape: (batch, pomo)
        # self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # # shape: (batch, pomo)
        # self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # # shape: (batch, pomo)
        # self.current_coord = self.depot_node_xy[:, :1, :]  # depot
        # # shape: (batch, pomo, 2)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.mask = torch.where(torch.sum(self.ninf_mask, dim = -1) == self.problem_size, float('-inf'), 0.0).to(self.device)

        # print("self.step_state.mask: ", self.step_state.mask[0])

        # self.step_state.finished = self.finished
        # self.step_state.current_time = self.current_time
        # self.step_state.length = self.length
        # self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # print("mask: ", self.step_state.mask)
        # print("selected: ", selected)

        # Dynamic-1
        ####################################
        self.current_node = selected #shape (batch, pomo)
        # self.current_route = s #shape(batch, pomo)
        self.old_routes = self.step_state.current_routes.clone()
        self.routes_cost  = self.step_state.cur_travel_time_routes.clone()
        self.old_truck = self.step_state.truck_num.clone()
        self.old_demand = self.step_state.cur_demands.clone()
        self.old_mask = self.step_state.ninf_mask.clone()
        self.node_mask = torch.gather(self.step_state.ninf_mask, 2,self.current_node.unsqueeze(-1).unsqueeze(-1).expand(-1,-1, 1, self.problem_size)).squeeze(2)
        self.step_state.current_routes, self.step_state.cur_travel_time_routes, self.step_state.truck_num, self.step_state.cur_demands, self.step_state.cur_index, self.current_route = update_route(self.node_xy, self.depot, self.node_demand, self.service_time, self.step_state.cur_demands, self.step_state.truck_num, self.step_state.current_routes, self.step_state.cur_travel_time_routes, self.current_node,self.node_mask, self.speed)

        # print("cur route: ", self.current_route)
        # print("step state cur route: ", self.step_state.current_routes)
        # print("cost: ", torch.sum(self.step_state.cur_travel_time_routes, dim=-1))
        # print("demand: ", self.step_state.cur_demands)
        
        
        # Mask

        self.init_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node, :] = 1 
        self.mask_node[self.BATCH_IDX, self.POMO_IDX, self.current_node] = 1

        self.step_state.init_mask = self.init_mask
        
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, :, self.current_route] = self.init_mask[self.BATCH_IDX, self.POMO_IDX, :, self.current_route]
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node, :] = 1

       
        self.step_state.ninf_mask = update_demand_mask(self.step_state.cur_demands, 
                                                       self.node_demand, 
                                                       self.step_state.ninf_mask, 
                                                       self.max_demand,
                                                       self.current_route)

        #route limit constraint
        self.step_state.ninf_mask = update_duration_mask(self.step_state.ninf_mask, 
                                                         self.step_state.current_routes,
                                                         self.step_state.cur_travel_time_routes,
                                                         self.service_time, 
                                                         self.route_limit,
                                                         self.step_state.truck_num,
                                                         self.current_route, 
                                                         self.node_xy,
                                                         self.depot, 
                                                         self.speed)
        self.c_mask = torch.sum(self.step_state.ninf_mask, dim = -1)
        
        self.mask = torch.where(self.c_mask == self.problem_size, float('-inf'), 0.0)


        self.step_state.mask = self.mask 

        self.c_mask = torch.sum(self.c_mask, dim=-1).squeeze(1)
        indices = (self.c_mask == self.problem_size * self.problem_size).nonzero()
        if indices.size(0) > 0:
            k = indices.size(0)
            selected = self.current_node[indices] # shape(k, pomo)
            enode_demand = self.node_demand[indices]
            
            etruck_num  = self.old_truck[indices]
            eroutes = self.old_routes[indices]
            ecost = self.routes_cost[indices]
            edemand = self.old_demand[indices]
            emask = self.old_mask[indices]
            enodemask = self.mask_node[indices]
            zero_indices = (etruck_num == 0).argmax(dim=2)  # shape (k,1)
            self.K_IDX = torch.arange(k)[:, None].expand(k, self.pomo_size).to(self.device)
            self.POMO_IDX1 = torch.arange(self.pomo_size)[None, :].expand(k, self.pomo_size).to(self.device)
            etruck_num[ self.K_IDX, self.POMO_IDX1, zero_indices] = 1
            eroutes[self.K_IDX, self.POMO_IDX1, zero_indices, :] = selected
            select_demand = torch.gather(enode_demand.unsqueeze(1).expand(-1, self.pomo_size, -1),2,  selected.unsqueeze(-1))
            edemand[ self.K_IDX, self.POMO_IDX1, zero_indices] = select_demand.squeeze(-1)
            selected_node_xy = torch.gather(self.node_xy[indices].unsqueeze(-1, self.pomo_size, -1, -1), 2, selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2))  # Shape: (batch, m, 2)
            selected_service_time= torch.gather(self.service_time[indices].unsqueeze(-1, self.pomo_size, -1), 2, selected.unsqueeze(-1).expand(-1, -1, 1)).squeeze(-1) 
            
            # Compute the difference in coordinates
            delta = selected_node_xy - self.depot[indices].unsqueeze(1)  # Shape: (batch, m, 1, 2)
        
            # Compute Euclidean distances
            distances = torch.sqrt(torch.sum(delta ** 2, dim=-1)).squeeze(-1)  # Shape: (batch, m, 1)
        
            # Compute travel time by dividing distance by speed
            travel_times = distances / self.speed 
            ecost[ self.K_IDX, self.POMO_IDX1, zero_indices, 0] =  travel_times + selected_service_time

            ecost[ self.K_IDX, self.POMO_IDX1, zero_indices, 1] =  travel_times 

            emask[ self.K_IDX, self.POMO_IDX1, :, zero_indices] = 0
            emask = emask | (enodemask.unsqueeze(-1).expand(-1, -1, -1, self.problem_size))
            emask = update_demand_mask(edemand, 
                                        self.node_demand[indices], 
                                        emask, 
                                        self.max_demand[indices],
                                        zero_indices)
            emask = update_duration_mask(emask, 
                                        eroutes,
                                        ecost,
                                        self.service_time[indices], 
                                        self.route_limit[indices],
                                        etruck_num,
                                        zero_indices, 
                                        self.node_xy[indices],
                                        self.depot[indices], 
                                        self.speed)

            self.old_truck[indices] = etruck_num  
            self.old_routes[indices] = eroutes 
            self.routes_cost[indices] = ecost 
            self.old_demand[indices] = edemand
            self.old_mask[indices] = emask
            self.step_state.current_routes = self.old_routes
            self.step_state.cur_travel_time_routes = self.routes_cost  
            self.step_state.truck_num = self.old_truck
            self.step_state.cur_demands = self.old_demand
            self.step_state.ninf_mask = self.old_mask 
            self.c_mask = torch.sum(self.step_state.ninf_mask, dim = -1)
        
            self.mask = torch.where(self.c_mask == self.problem_size, float('-inf'), 0.0)


            self.step_state.mask = self.mask 


            

        # indices = (self.c_mask == self.problem_size * self.problem_size).nonzero()
        
        # if indices.size(0) > 0:
        #     temp_data = [self.problem_data[i][indices[0]] for i in range(11)]
        #     print()
        #     for data in temp_data:
        #         print(data)
        
        # print("mask 2: ", self.step_state.mask)

        # returning values
        done = self.step_state.done
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        travel_distances = torch.sum(self.step_state.cur_travel_time_routes, -1)
        travel_distances = torch.sum(travel_distances, dim = -1) # shape (batch, pomo)
        # gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # # shape: (batch, pomo, selected_list_length, 2)
        # all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # # shape: (batch, pomo, problem+1, 2)

        # ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # # shape: (batch, pomo, selected_list_length, 2)

        # rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        # segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # # shape: (batch, pomo, selected_list_length)

        # # if self.loc_scaler:
        # #     segment_lengths = torch.round(segment_lengths * self.loc_scaler) / self.loc_scaler

        # travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def generate_dataset(self, num_samples, problem_size, path):
        data = self.get_random_problems(num_samples, problem_size, normalized=False)
        dataset = [attr.cpu().tolist() for attr in data]
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(path, 'wb') as f:
            pickle.dump(list(zip(*dataset)), f, pickle.HIGHEST_PROTOCOL)
        print("Save VRPLTW dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=1000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data], [i[4] for i in data], [i[5] for i in data], [i[6] for i in data], [i[7] for i in data]
        depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor(route_limit), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)
        node_demand = node_demand / capacity.view(-1, 1)
        data = (depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end)
        return data

    def load_pkl_data(self):
        # Load train data
        with open(self.train_pkl_path, 'rb') as pkl_file:
            train_data = pickle.load(pkl_file)
        with open(self.train_pkl_path.replace(".pkl", ".json"), 'r') as meta_file:
            self.train_ntruck_category = json.load(meta_file)

        num_instances = len(train_data[0])
        tw_start = torch.zeros(size=(num_instances, 1)).to(self.device)
        tw_end = torch.zeros(size=(num_instances, 1)).to(self.device)

        # pickle_data = [depot_xy, node_xy, node_demand, route_limit, service_time, init_routes, mask_node, truck_num, capacity, solomon_cost]

        self.train_dataset = [
                    train_data[0].squeeze(1).to(self.device),
                    train_data[1].to(self.device),
                    train_data[2].to(self.device),
                    train_data[3].to(self.device),
                    train_data[4].to(self.device),
                    tw_start,
                    tw_end,
                    train_data[5].clone().detach().to(dtype=torch.int64).to(self.device),
                    train_data[6].clone().detach().to(dtype=torch.int64).to(self.device),
                    train_data[7].clone().detach().to(dtype=torch.int64).to(self.device),
                    train_data[8].to(self.device),
                    train_data[9].to(self.device) ]
        
        # Load val data
        with open(self.val_pkl_path, 'rb') as pkl_file:
            val_data = pickle.load(pkl_file)
        with open(self.val_pkl_path.replace(".pkl", ".json"), 'r') as meta_file:
            self.val_ntruck_category = json.load(meta_file)

        num_instances = len(val_data[0])
        tw_start = torch.zeros(size=(num_instances, 1)).to(self.device)
        tw_end = torch.zeros(size=(num_instances, 1)).to(self.device)

        self.val_dataset = [
                    val_data[0].squeeze(1).to(self.device), # depot_xy
                    val_data[1].to(self.device), # node_xy
                    val_data[2].to(self.device), # node_demand
                    val_data[3].to(self.device), # route_limit
                    val_data[4].to(self.device), # service_time
                    tw_start,
                    tw_end,
                    val_data[5].clone().detach().to(dtype=torch.int64).to(self.device), # init_routes
                    val_data[6].clone().detach().to(dtype=torch.int64).to(self.device), # mask_node
                    val_data[7].clone().detach().to(dtype=torch.int64).to(self.device), # truck_num
                    val_data[8].to(self.device), # capacity
                    val_data[9].to(self.device) ] # solomon_cost
 

    def get_batch_data(self, mode, batch_size):
        if mode == "train":
            ntruck = random.choice(list(self.train_ntruck_category.keys()))
            batch_idx = random.sample(self.train_ntruck_category[ntruck], batch_size)

            batch_data = tuple(torch.stack([self.train_dataset[i][idx] for idx in batch_idx]) for i in range(len(self.train_dataset)))
        
        else:
            ntruck = random.choice(list((self.val_ntruck_category).keys()))
            batch_idx = random.sample(self.val_ntruck_category[ntruck], batch_size)

            batch_data = tuple(torch.stack([self.val_dataset[i][idx] for idx in batch_idx]) for i in range(len(self.val_dataset)))

        print(">>>>>>> ntruck ", ntruck)
        
        return (batch_data, int(ntruck))
    
    def get_random_problems(self, mode, batch_size):
        # if self.train_dataset is None:
        #     self.load_pkl_data()
        # return self.get_batch_data(mode, batch_size)    
        data = [
            torch.tensor([[0.1294, 0.4987]]),
            torch.tensor([[[0.1805, 0.3714],
                     [0.0928, 0.2687],
                     [0.0677, 0.9520],
                     [0.6354, 0.3108],
                     [0.8685, 0.9355],
                     [0.7596, 0.9022],
                     [0.9331, 0.1802],
                     [0.8760, 0.7106],
                     [0.9255, 0.8798],
                     [0.8981, 0.4332],
                     [0.9541, 0.8274],
                     [0.7320, 0.4478],
                     [0.2886, 0.9342],
                     [0.1714, 0.1963],
                     [0.2627, 0.2424],
                     [0.7302, 0.3479],
                     [0.0654, 0.5889],
                     [0.3937, 0.8577],
                     [0.0075, 0.9549],
                     [0.8513, 0.4495],
                     [0.9873, 0.9746],
                     [0.5572, 0.7557],
                     [0.2906, 0.7220],
                     [0.1334, 0.3847],
                     [0.8979, 0.6113],
                     [0.8266, 0.5148],
                     [0.2219, 0.4461],
                     [0.1626, 0.3837],
                     [0.3395, 0.7275],
                     [0.1854, 0.6380],
                     [0.1392, 0.9267],
                     [0.6791, 0.6457],
                     [0.6133, 0.2108],
                     [0.9650, 0.2945],
                     [0.1393, 0.2767],
                     [0.9385, 0.1725],
                     [0.0746, 0.0277],
                     [0.9909, 0.1188],
                     [0.9848, 0.2015],
                     [0.9495, 0.9896],
                     [0.0401, 0.3935],
                     [0.8824, 0.0574],
                     [0.2531, 0.4751],
                     [0.0828, 0.2982],
                     [0.0851, 0.3392],
                     [0.6903, 0.2218],
                     [0.5135, 0.7989],
                     [0.3205, 0.2812],
                     [0.9034, 0.1054],
                     [0.9585, 0.3415]]]),
            torch.tensor([[0.2000, 0.0750, 0.2250, 0.1750, 0.1750, 0.2250, 0.0250, 0.0500, 0.1500,
                     0.1000, 0.1250, 0.0750, 0.1250, 0.0250, 0.1500, 0.1000, 0.1500, 0.2250,
                     0.1500, 0.0500, 0.0500, 0.0500, 0.0250, 0.1000, 0.2000, 0.0750, 0.1250,
                     0.1000, 0.1500, 0.0500, 0.1750, 0.0750, 0.0250, 0.0250, 0.1500, 0.1750,
                     0.2000, 0.0500, 0.0750, 0.1250, 0.1250, 0.1000, 0.1250, 0.1500, 0.1000,
                     0.2250, 0.2000, 0.2250, 0.0500, 0.1750]]),
            torch.tensor([3.]),
            torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0.]]),
            torch.tensor([[0.]]),
            torch.tensor([[0.]]),
            torch.tensor([[22,  7, 15, 25, 37, 38, 20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]),
            torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0]]),
            torch.tensor([[5, 4, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0]]),
            torch.tensor([1.]),
            torch.tensor([100.])
        ]

        return (tuple(data), 7)



    def augment_xy_data_by_8_fold(self, xy_data):
        # xy_data.shape: (batch, N, 2)

        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        # x,y shape: (batch, N, 1)

        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape: (8*batch, N, 2)

        return aug_xy_data


    