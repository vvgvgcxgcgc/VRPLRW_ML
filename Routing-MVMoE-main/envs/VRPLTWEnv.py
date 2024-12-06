from dataclasses import dataclass
import torch
import os, pickle
import numpy as np

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




    max_route: int = 0

def update_duration_mask(cur_mask, cur_routes, routes_cost, service_time,  max_duration, truck_num, selected_route, cur_index, node_xy, depot, speed = 1.0):
    batch, pomo = cur_index.shape()
    BATCH_IDX = torch.arange(batch)[:, None].expand(batch, pomo)
    POMO_IDX = torch.arange(pomo)[None, :].expand(batch, pomo)
    selected_route_cost = torch.gather(routes_cost, 2, selected_route.unsqueeze(-1).unsqueeze(-1)).squeeze(2)
    nodes_route = torch.gather(cur_routes, 2, selected_route.unsqueeze(-1).unsqueeze(-1)).squeeze(2)
    nodes_num = torch.gather(cur_routes, 2, selected_route.unsqueeze(-1)).squeeze(2)
    total_cost = torch.sum(selected_route_cost, dim = 2)
    suspect_pos = torch.concat((nodes_route[BATCH_IDX, POMO_IDX,cur_index].unsqueeze(-1), nodes_route[BATCH_IDX, POMO_IDX,cur_index -1].unsqueeze(-1), nodes_route[BATCH_IDX, POMO_IDX,cur_index -1].unsqueeze(-1)), 2)
    node_xy_expand = node_xy.unqueeze(1).expand(-1, pomo, -1, -1)
    nodes_route_coords = torch.gather(node_xy_expand, 2, suspect_pos.unsqueeze(-1).expand(-1, -1, -1, 2)) #shape (batch, pomo, 3, 2)
    truck_num = truck_num - 1
    nodes_route_coords[:, :, 2, :] = torch.where(cur_index.unsqueeze(-1).expand(-1, -1, 2) < truck_num, nodes_route_coords[:, :, 2, :], depot.unsqueeze(1).expand(-1, pomo, -1))
    distances= torch.sqrt(torch.sum((nodes_route_coords.unsqueeze(3) - node_xy_expand.unsqueeze(2)) ** 2, dim=-1))/speed
    check = (distances[:, :, :, 0] + distances[:, :, :, 1] + service_time.unsqueeze(1).expand(-1, pomo, -1) - selected_route_cost[BATCH_IDX, POMO_IDX,cur_index].unsqueeze(-1) +total_cost.unsqueeze(-1)) > max_duration.unsqeeze(-1).unsqueeze(-1)
    check += (distances[:, :, :, 1] + distances[:, :, :, 2] + service_time.unsqueeze(1).expand(-1, pomo, -1) - selected_route_cost[BATCH_IDX, POMO_IDX,cur_index].unsqueeze(-1) +total_cost.unsqueeze(-1)) > max_duration.unsqeeze(-1).unsqueeze(-1)
    check = check <  2
    cur_mask[BATCH_IDX, POMO_IDX, :,  selected_route] = torch.where(cur_mask[BATCH_IDX, POMO_IDX, :,  selected_route] == 1, check, 0)

    return cur_mask



def update_demand_mask(cur_demands, node_demand, cur_mask, max_demand, selected_route):
    '''
        cur_demands Tensor shape (batch, pomo, n)
        node_demand Tensor shape (batch, n)
        cur_mask Tensor shape (batch, pomo, n, n)
        max_demand Tensor shape(batch)
    '''
    batch,pomo,n = cur_demands.shape()

    BATCH_IDX = torch.arange(batch)[:, None].expand(batch, pomo)
    POMO_IDX = torch.arange(pomo)[None, :].expand(batch, pomo)
    selected_route_demand = torch.gather(cur_demands, 2, selected_route.unsqueeze(-1))
    node_demand = node_demand.unsqueeze(1).expand(-1, pomo, n)
    round_error_epsilon = 0.00001
    max_demand = max_demand.unsqueeze(-1).unsqueeze(-1)
    demand_mask = selected_route_demand + node_demand + round_error_epsilon > max_demand #shape(batch, pomo, n)
    cur_mask[BATCH_IDX, POMO_IDX, :,  selected_route] = torch.where(cur_mask[BATCH_IDX, POMO_IDX, :,  selected_route] == 0, demand_mask, 1)
    return cur_mask

    
def update_route(node_xy, depot,node_demand,service_time, cur_demands, truck_num, cur_routes,routes_cost,  selected_node,node_mask, speed = 1.0):
    '''
    Update routes each decode time
    node_xy: Tensor shape (batch, n, 2) contains coord of n customer
    depot: Tensor shape (batch, 2)     contains coord of depot point
    truck_num: Tensor shape (batch, pomo, n) contains  number of customer in route, maximize n route, if just use m < n route, padding 0
    cur_routes: Tensor shape (batch, pomo, n, n) each routes contains index of customer, padding 0
    routes_cost: Tensor shape (batch, pomo, n, n + 1) cost of from right previous of each customer, add value from last node to depot, padding 0
    selected_node: Tensor shape(batch, pomo) contains customer index chosen by model
    selected_route: Tensor shape(batch, pomo) contain route index chosen by model
    Return:
        new_routes // same as cur_routes, but is inserted selected customer into selected route
        shape(batch, pomo, n, n)
        new_routes_cost shape(batch, pomo, n, n + 1)

    '''
    batch, pomo, _, n = cur_routes.shape()
    BATCH_IDX = torch.arange(batch)[:, None, None].expand(batch, pomo, n)
    POMO_IDX = torch.arange(pomo)[None, :, None].expand(batch, pomo, n)
    ROUTE_IDX =  torch.arange(pomo)[None, None, :].expand(batch, pomo, n)
    # raw_selected_route = selected_route.clone()
    selected_route = selected_route.unsqueeze(-1)
    node_num = torch.gather(truck_num, 2, selected_route) # shape (batch, pomo, 1)
    # node_num = node_num -1
    node_num = truck_num - 1 # shape (batch, pomo, n)
    # selected_route_expand = selected_route.unsqueeze(-1).expand(-1, -1, 1, n)
    # nodes_route = torch.gather(cur_routes, 2, selected_route_expand).squeeze(2)
    nodes_route = cur_routes # shape (batch, pomo, n, n)

    node_xy_expand = node_xy.unqueeze(1).unsqueeze(2).expand(-1, pomo,n, -1, -1)
    raw_selected_node = selected_node.clone()
    selected_node = selected_node.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    selected_node_coord = torch.gather(node_xy_expand, 2, selected_node).unsqueeze(2).expand(-1, -1, n, -1, -1) # shape( batch, pomo, 1, 2)
    nodes_route_expand = nodes_route.unsqueeze(-1).expand(-1,-1,-1, -1, 2) #shape (batch, pomo, n, n, 2 )
    nodes_route_coords = torch.gather(node_xy_expand, 3, nodes_route_expand) #shape (batch, pomo, n, n, 2)
    node_num_expand = node_num.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1,1, 2) # shape(batch, pomo, 1, 2)
    last_node_coord = torch.gather(nodes_route_coords, 3, node_num_expand) #shape(batch, pomo,n,1, 2)
    depot_expand = depot.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, pomo, n, 1, -1) #shape(batch, pomo,n,1, 2)
    distance_lastnode_depot = torch.sqrt(torch.sum((last_node_coord - depot_expand) ** 2, dim=-1))/speed
    #shape(batch, pomo,n, 1)
    distance_selected_depot = torch.sqrt(torch.sum((selected_node_coord - depot_expand) ** 2, dim=-1))/speed
    #shape(batch, pomo,n, 1)
    distances = torch.sqrt(torch.sum((selected_node_coord - nodes_route_coords) ** 2, dim=-1)) / speed
    # shape(batch, pomo,n,  n)

    range_tensor = torch.arange(n).unsqueeze(0).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 1, n)
    
    padding = range_tensor > node_num.unsqueeze(-1)
    padding = padding*10 #shape (batch, pomo, n, n)
    distances = distances + padding

    new_distances = distances.clone()
    new_distances = torch.concat((distance_selected_depot, new_distances), dim = 3) # shape (batch, pomo, n, n+1)
    node_num = node_num + 1
    distances[BATCH_IDX, POMO_IDX,ROUTE_IDX, node_num] = distance_lastnode_depot
    distance_lastnode_depot = distance_lastnode_depot*10
    distances = torch.concat((distances,distance_lastnode_depot ), dim = 3) # shape (batch, pomo, n + 1)
    # lost_index = raw_selected_route.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n + 1)
    # lost_distance = torch.gather(routes_cost, 2,lost_index ).squeeze(2) #shape(batch, pomo, n+1)
    inserted_cost = new_distances + distances - routes_cost #shape (batch, pomo,n, n + 1)
    node_mask = node_mask*20
    inserted_cost = inserted_cost + node_mask.unsqueeze(-1)
    min_indices = torch.argmin(inserted_cost.view(batch,pomo, -1), dim=2)
    selected_route_index = min_indices // inserted_cost.size(-1) # shape(batch, pomo)
    insert_index = min_indices % inserted_cost.size(-1) # shape (batch, pomo)
    # insert_index = torch.argmin(inserted_cost, dim = 3) # shape (batch, pomo, )
    selected_node_route = torch.gather(nodes_route, 2,selected_route_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n)).squeeze(2)
    lost_distance = torch.gather(routes_cost, 2, selected_route_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n + 1)).squeeze(2)
    selected_dis = torch.gather(distances, 2, selected_route_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n + 1)).squeeze(2)
    selected_new_dis = torch.gather(new_distances, 2, selected_route_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, n + 1)).squeeze(2)
    new_nodes_route = torch.zeros((batch,pomo, n + 1))
    expanded_indices = torch.arange(n + 1).expand(batch,pomo, -1)  # Shape: (batch, n + 1)
    mask_before = expanded_indices < insert_index.unsqueeze(1)  # Shape: (batch, n+1)
    mask_after = expanded_indices > insert_index.unsqueeze(1)
    new_nodes_route[mask_before] = selected_node_route[mask_before[:, :, :-1]]
    new_nodes_route[mask_after] = selected_node_route[mask_after[:, :, 1:]]
    new_nodes_route[BATCH_IDX, POMO_IDX,  insert_index] = raw_selected_node
    cur_routes[BATCH_IDX, POMO_IDX, selected_route_index] = new_nodes_route[:, :, :-1]
    new_route_cost = torch.zeros((batch, pomo, n + 2))
    expanded_indices = torch.arange(n + 2).expand(batch,pomo, -1)  # Shape: (batch, n + 2)
    mask_before = expanded_indices < insert_index.unsqueeze(1)  # Shape: (batch, n+2)
    mask_after = expanded_indices > insert_index.unsqueeze(1)
    new_route_cost[mask_before] = lost_distance[mask_before[:, :, :-1]]
    lost_distance[BATCH_IDX, POMO_IDX, insert_index] = selected_dis[BATCH_IDX, POMO_IDX, insert_index] 
    new_route_cost[mask_after] = lost_distance[mask_after[:, :, 1:]]
    new_route_cost[BATCH_IDX, POMO_IDX, insert_index] = selected_new_dis[BATCH_IDX, POMO_IDX, insert_index] +  service_time.unsqueeze(1).expand(-1, pomo, -1)[BATCH_IDX, POMO_IDX, raw_selected_node]
    routes_cost[BATCH_IDX, POMO_IDX, selected_route_index] = new_route_cost[:, :, :-1]
    truck_num[BATCH_IDX, POMO_IDX, selected_route_index]  += 1


    node_demand = node_demand.unsqueeze(1).expand(-1, pomo, -1)
    cur_demands[BATCH_IDX, POMO_IDX, selected_route_index] += node_demand[BATCH_IDX, POMO_IDX, raw_selected_node]

    return cur_routes, routes_cost, truck_num, cur_demands, insert_index,selected_route_index





def init_travel_time_demand(depot_xy, node_xy, node_demand, service_time,  node_indices, truck_num, speed=1.0):
    """
    Computes the travel time from the depot to a set of nodes identified by indices.

    Args:
        depot_xy (torch.Tensor): Tensor of shape (batch, 1, 2) representing depot coordinates.
        node_xy (torch.Tensor): Tensor of shape (batch, n, 2) representing coordinates of all nodes.
        node_demand: Tensor of shape(batch, n)
        node_indices (torch.Tensor): Tensor of shape (batch, m) containing indices of nodes to calculate travel time for.
        truck_num: Tensor shape (batch, n)
        speed (float): Travel speed, default is 1.0.

    Returns:
        torch.Tensor: Tensor of shape (batch, m) representing travel times from depot to each selected node.
    """
    # Gather node coordinates using the indices
    routes_demand = torch.gather(node_demand, 1, node_indices)
    routes_demand = routes_demand * truck_num

    selected_node_xy = torch.gather(node_xy, 1, node_indices.unsqueeze(-1).expand(-1, -1, 2))  # Shape: (batch, m, 2)
    selected_service_time= torch.gather(service_time, 1, node_indices )
    # Compute the difference in coordinates
    delta = selected_node_xy - depot_xy  # Shape: (batch, m, 2)

    # Compute Euclidean distances
    distances = torch.sqrt(torch.sum(delta ** 2, dim=-1))  # Shape: (batch, m)

    # Compute travel time by dividing distance by speed
    travel_times = distances / speed 
    batch, m = travel_times.shape

    # Apply mask to travel_time
    travel_times = travel_times * truck_num  # Masked values are set to 0
    
    n = node_xy.size(1)

    # Initialize result matrix with zeros
    travel_matrix = torch.zeros((batch, n, n + 1), device=travel_times.device)

    # Fill the first m values of each row in the n x n matrix
    travel_matrix[:, :m, 0] = travel_times + selected_service_time
    travel_matrix[:, :m, 1] = travel_times

    routes_matrix = torch.zeros((batch, n, n ), device=travel_times.device)
    routes_matrix[:, :m, 0] = node_indices

   
    return travel_matrix, routes_matrix, routes_demand
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

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        self.current_time = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.open = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, problems=None, aug_factor=1):
        if problems is not None:
            depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, init_routes, mask_node, truck_num, max_demand = problems
        else:
            depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end, init_routes, mask_node, truck_num, max_demand = self.get_random_problems(batch_size, self.problem_size, normalized=True)
        self.batch_size = depot_xy.size(0)
        self.num_problem = depot_xy.size(1)
        route_limit = route_limit[:, None] if route_limit.dim() == 1 else route_limit

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = self.augment_xy_data_by_8_fold(depot_xy)
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                route_limit = route_limit.repeat(8, 1)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
            else:
                raise NotImplementedError
        self.node_xy = node_xy
        self.depot = depot_xy
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)

        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        depot_service_time = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        depot_tw_start = torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_start
        depot_tw_end = torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_end
        # shape: (batch, 1)
        self.node_demand = node_demand #shape (batch, n)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        self.max_demand = max_demand
        # shape: (batch,)
        self.route_limit = route_limit
        self.init_routes = init_routes #shape(batch, pomo, n)
        # shape: (batch, 1)
        self.depot_node_service_time = torch.cat((depot_service_time, service_time), dim=1)
        self.service_time = service_time
        # shape: (batch, problem+1)
        self.depot_node_tw_start = torch.cat((depot_tw_start, tw_start), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_tw_end = torch.cat((depot_tw_end, tw_end), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)
        self.truck_num= truck_num

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_service_time = service_time
        self.reset_state.node_tw_start = tw_start
        self.reset_state.node_tw_end = tw_end
        self.reset_state.prob_emb = torch.FloatTensor([1, 0, 0, 1, 1]).unsqueeze(0).to(self.device)  # bit vector for [C, O, B, L, TW]

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        # self.step_state.open = torch.zeros(self.batch_size, self.pomo_size).to(self.device)
        # self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size + 1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.step_state.PROBLEM = self.problem
        self.step_state.truck_num = self.truck_num.unsqueeze(1).expand(self.batch_size, self.pomo_size,-1)
        
        # padded_routes = torch.full((self.batch_size, self.problem_size), self.problem_size, dtype=init_routes.dtype)  # Shape: (2, 5)
        # padded_routes[:,0] =  init_routes # init_routes.shape(batch,problem)
        # init_routes = init_routes.expand(-1, self.pomo_size, -1)
        # self.step_state.current_routes = init_routes
        self.step_state.cur_travel_time_routes, self.step_state.current_routes,  self.step_state.cur_demands = init_travel_time_demand(depot_xy, node_xy,node_demand, init_routes, self.truck_num, self.speed)
        self.step_state.cur_travel_time_routes = self.step_state.cur_travel_time_routes.unsqueeze(1).expand(-1, self.pomo_size, -1, -1)
        self.step_state.current_routes = self.step_state.current_routes.unsqueeze(1).expand(-1, self.pomo_size, -1, -1)
        self.step_state.cur_demands = self.step_state.cur_demands.unsqueeze(1).expand(-1, self.pomo_size, -1)
        # self.step_state.cur_demands = torch.where(init_routes == self.problem_size, torch.tensor(0, dtype=torch.int32), node_demand)
        # self.step_state.current_routes, self.step_state.cur_demands, self.step_state.tw_start_routes, self.step_state.tw_end_routes = calculate_tw_demand_index(tw_start, tw_end, node_demand, init_routes, self.truck_num, self.pomo_size)



    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.current_route = None
        #shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo)
        # self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size, self.problem_size)).to(self.device)
       

        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size,self.problem_size )).to(self.device)
        zero_mask = (self.step_state.truck_num == 0)  # Shape: (batch, pomo, m)
        zero_mask = zero_mask.unsqueeze(2).expand(self.batch_size, self.pomo_size, self.problem_size, -1)
        self.ninf_mask[zero_mask] = 1
        mask_node = mask_node.unsqueeze(1).unsqueeze(-1).expand(-1, self.pomo_size, self.problem_size, self.problem_size)
        self.ninf_mask[mask_node] = 1
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.current_coord = self.depot_node_xy[:, :1, :]  # depot
        # shape: (batch, pomo, 2)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        # self.step_state.finished = self.finished
        # self.step_state.current_time = self.current_time
        # self.step_state.length = self.length
        # self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected #shape (batch, pomo)
        # self.current_route = s #shape(batch, pomo)
        self.node_mask = torch.gather(self.ninf_mask, 2,self.current_node.unsqueeze(-1).unsqueeze(-1).expand(-1,-1, 1, self.problem_size)).squeeze(2)
        self.step_state.current_routes, self.step_state.cur_travel_time_routes, self.step_state.truck_num, self.step_state.cur_demands, self.step_state.cur_index, self.current_route  = update_route(self.node_xy, self.depot, self.node_demand,self.service_time, self.step_state.cur_demands, self.step_state.truck_num,self.step_state.current_routes, self.step_state.cur_travel_time_routes, self.current_node,self.node_mask, self.speed)



        # demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # # shape: (batch, pomo, problem+1)
        # gathering_index = selected[:, :, None]
        # # shape: (batch, pomo, 1)
        # selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # # shape: (batch, pomo)
        # self.load -= selected_demand
        # # shape: (batch, pomo)
        # self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # # shape: (batch, pomo, 0~)

        # # Dynamic-2
        # ####################################
        # self.at_the_depot = (selected == 0)

        # demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # # shape: (batch, pomo, problem+1)
        # gathering_index = selected[:, :, None]
        # # shape: (batch, pomo, 1)
        # selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # # shape: (batch, pomo)
        # self.load -= selected_demand
        # self.load[self.at_the_depot] = 1  # refill loaded at the depot

        # current_coord = self.depot_node_xy[torch.arange(self.batch_size)[:, None], selected]
        # # shape: (batch, pomo, 2)
        # new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        # # shape: (batch, pomo)
        # self.length = self.length + new_length
        # self.length[self.at_the_depot] = 0  # reset the length of route at the depot
        # self.current_coord = current_coord

        # Mask
        self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node, :] = 1
       
        self.ninf_mask = update_demand_mask(self.step_state.cur_demands, self.node_demand,self.ninf_mask, self.max_demand,self.current_route)

        #route limit constraint
        self.ninf_mask = update_duration_mask(self.ninf_mask, self.step_state.current_routes,self.step_state.cur_travel_time_routes,self.service_time, self.route_limit,self.step_state.truck_num,self.current_route,self.step_state.cur_index, self.node_xy,self.depot, self.speed)
        self.mask = torch.sum(self.ninf_mask, dim = -1)
        self.mask = torch.where(self.mask == self.pomo_size, float('-inf'), 0)

        self.step_state.selected_count = self.selected_count

        self.step_state.ninf_mask = self.mask

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

    def get_random_problems(self, batch_size, problem_size, normalized=True):
        depot_xy = torch.rand(size=(batch_size, 1, 2))  # (batch, 1, 2)
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

        if problem_size == 20:
            demand_scaler = 30
        elif problem_size == 50:
            demand_scaler = 40
        elif problem_size == 100:
            demand_scaler = 50
        elif problem_size == 200:
            demand_scaler = 70
        else:
            raise NotImplementedError

        route_limit = torch.ones(batch_size) * 3.0

        # time windows (vehicle speed = 1.):
        #   1. The setting of "MTL for Routing Problem with Zero-Shot Generalization".
        """
        self.depot_start, self.depot_end = 0., 4.6.
        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, problem_size)
        tw_length = b + (c - b) * torch.rand(batch_size, problem_size)
        c = (node_xy - depot_xy).norm(p=2, dim=-1)
        h_max = (self.depot_end - service_time - tw_length) / c * self.speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, problem_size)) * c / self.speed
        tw_end = tw_start + tw_length
        """
        #   2. See "Learning to Delegate for Large-scale Vehicle Routing" in NeurIPS 2021.
        #   Note: this setting follows a similar procedure as in Solomon, and therefore is more realistic and harder.
        service_time = torch.ones(batch_size, problem_size) * 0.2
        travel_time = (node_xy - depot_xy).norm(p=2, dim=-1) / self.speed
        a, b = self.depot_start + travel_time, self.depot_end - travel_time - service_time
        time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
        time_half_width = (service_time / 2 - self.depot_end / 3) * torch.rand(batch_size, problem_size) + self.depot_end / 3
        tw_start = torch.clamp(time_centers - time_half_width, min=self.depot_start, max=self.depot_end)
        tw_end = torch.clamp(time_centers + time_half_width, min=self.depot_start, max=self.depot_end)
        # shape: (batch, problem)

        # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
        instance_invalid, round_error_epsilon = False, 0.00001
        total_time = torch.max(0 + (depot_xy - node_xy).norm(p=2, dim=-1) / self.speed, tw_start) + service_time + (node_xy - depot_xy).norm(p=2, dim=-1) / self.speed > self.depot_end + round_error_epsilon
        # (batch, problem)
        instance_invalid = total_time.any()

        if instance_invalid:
            print(">> Invalid instances, Re-generating ...")
            return self.get_random_problems(batch_size, problem_size, normalized=normalized)
        elif normalized:
            node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)  # (batch, problem)
            return depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end
        else:
            node_demand = torch.Tensor(np.random.randint(1, 10, size=(batch_size, problem_size)))  # (unnormalized) shape: (batch, problem)
            capacity = torch.Tensor(np.full(batch_size, demand_scaler))
            return depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end

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


    
