class Customer_Tu:
    def __init__(self, demand):
        self.demand = demand # q

class Route_Tu:
    def __init__(self):
        self.customers = []
        self.current_capacity = 0
        self.current_distance = 0

def nearest_neighbor(customers, m, Q, D, cost_matrix):
    COST = 0
    n = len(customers)

    unrouted_customers = set(range(n))
    routes = []

    # each route
    for vehicle in range(m):

        # print(f'>>>>> Unrouted customers for route {vehicle}: {unrouted_customers}')

        # if no more customers left today
        if not unrouted_customers:
            break

        route = Route_Tu()
        # current_time = e0
        last_customer_index = -1
        nearest_customer = -1
        delays = []
        
        while True:
            best_cost = float('inf') #duong vc
            for j in unrouted_customers:
                if is_feasible(route, j, customers, Q, D, cost_matrix):

                    costt = cost_matrix[last_customer_index + 1][j + 1]

                    cost = 0.5 * cost_matrix[j + 1][0] + \
                        0.5 * (costt)

                    if cost < best_cost:
                        best_cost = cost
                        nearest_customer = j

            print(f'Nearest: {nearest_customer}')
            
            if nearest_customer == -1 or nearest_customer not in unrouted_customers:  
                if route.customers != []:
                    COST += route.current_capacity
                    print(f'End route!')
                    print()
                    break
                else:
                    print('No more available customers!')
                    print()
                    return -1, -1
            
            route.customers.append(nearest_customer)
            route.current_capacity += customers[nearest_customer].demand
            route.current_distance += cost_matrix[last_customer_index + 1][nearest_customer + 1]
            last_customer_index = nearest_customer
            
            unrouted_customers.remove(nearest_customer)
            print(f"Unrouted customers: {unrouted_customers}")
            print(f"Route: {route.customers}")
            print()

        if route.customers != []:
            routes.append(route.customers)

    if not unrouted_customers:
        return routes
    else:
        return -1


def is_feasible(route, customer_index, customers, Q, D, cost_matrix):
    customer = customers[customer_index]
    # print(customer.demand)
    current_demand = route.current_capacity + customer.demand
    
    if route.customers != []:
        if route.current_distance + cost_matrix[route.customers[-1] + 1][customer_index + 1] + cost_matrix[customer_index + 1][0] > D:
            # print(f"Try {customer_index}: Distance exceeded!")
            return False
    elif route.current_distance + cost_matrix[0][customer_index + 1] + cost_matrix[customer_index + 1][0] > D:
        # print(f"Try {customer_index}: Distance exceeded!")
        return False
    
    if current_demand > Q:
        # print(f"Try {customer_index}: Demand exceeded!")
        return False
        
    return True



##### READ INPUT #####



test_dir = "C:/Tu/STUDY/4/TienHoa/BTL/gen_data/test1"

data_infos = []
with open(f"{test_dir}/data_info.txt", mode = "r") as datainfo_file:
    data_infos = datainfo_file.readlines()

D, Q = [int(float(c)) for c in data_infos[0].split(" ")]

customers_info = []
with open(f"{test_dir}/customer_data.txt", mode="r") as customerinfo_file:
    customers_info = customerinfo_file.readlines()

m = n = len(customers_info)
customers = []
for i in range(n):
    qi = int(float(customers_info[i]))
    customers.append(Customer_Tu(qi))

# print(customers[0].demand)
# assert False

cost_data = []
with open(f"{test_dir}/cost_matrix.txt", "r") as cost_file:
    cost_data = cost_file.readlines()

cost_matrix = []
for row in cost_data:
    row = [float(c) for c in row.split(" ")]
    cost_matrix.append(row)

routes, COST = nearest_neighbor(customers, m, Q, D, cost_matrix)

print()
print()
print(routes)
print(COST)
print()