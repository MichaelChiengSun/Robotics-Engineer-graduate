#!/usr/bin/env python
# # Artificial Intelligence: Sorting Algorithms

# ## 1. Represent tube data in graph
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue

# Visualisation of graph
def show_weighted_graph(networkx_graph, node_size, font_size, fig_size):
    # Allocate the given fig_size in order to have space for each node
    plt.figure(num=None, figsize=fig_size, dpi=80)
    plt.axis('off')
    # Compute the position of each vertex in order to display it nicely
    nodes_position = nx.spring_layout(networkx_graph) 
    # You can change the different layouts depending on your graph
    # Extract the weights corresponding to each edge in the graph
    edges_weights  = nx.get_edge_attributes(networkx_graph,'weight')
    # Draw the nodes (you can change the color)
    nx.draw_networkx_nodes(networkx_graph, nodes_position, node_size=node_size, node_color = ["orange"]*networkx_graph.number_of_nodes())
    # Draw only the edges
    colors = [networkx_graph[u][v]['color'] for u, v in networkx_graph.edges]
    nx.draw_networkx_edges(networkx_graph, nodes_position, edgelist=list(networkx_graph.edges), edge_color=colors, width=2)
    # Add the weights
    nx.draw_networkx_edge_labels(networkx_graph, nodes_position, edge_labels = edges_weights)
    # Add the labels of the nodes
    nx.draw_networkx_labels(networkx_graph, nodes_position, font_size=font_size, font_family='sans-serif')
    plt.axis('off')
    plt.show()

# Load data from csv file
df = pd.read_csv('tubedata.csv', header=None)
df_nx_graph = nx.Graph()
# Extract attributes from each row data
for row in range(df.shape[0]):
    starting_station = df.iloc[row,0]
    ending_station = df.iloc[row,1]
    tube_line = df.iloc[row,2]
    average_time_taken = df.iloc[row,3]
    main_zone = df.iloc[row,4]
    secondary_zone = df.iloc[row,5]
    df_nx_graph.add_node(starting_station, zone=main_zone)    # Starting station node

    if tube_line == "Bakerloo":
        color = 'brown'
    if tube_line == "Central":
        color = 'red'
    if tube_line == "Circle":
        color = 'yellow'
    if tube_line == "District":
        color = 'green'
    if tube_line == "East London":
        color = 'orange'
    if tube_line == "Hammersmith & City":
        color = 'pink'
    if tube_line == "Jubilee":
        color = 'lightgrey'
    if tube_line == "Metropolitan":
        color = 'purple'
    if tube_line == "Northern":
        color = 'black'
    if tube_line == "Piccadilly":
        color = 'blue'
    if tube_line == "Victoria":
        color = 'lightblue'
    if tube_line == "Waterloo & City":
        color = 'lime'
    # Ending station uses main zone if secondary zone is 0
    if secondary_zone == '0':
        # Ending station node
        df_nx_graph.add_node(ending_station, zone=main_zone)
        # Edge between two nodes
        df_nx_graph.add_edge(starting_station, ending_station, tube_line=[], weight=average_time_taken, color=color)
    else:
        df_nx_graph.add_node(ending_station, zone=secondary_zone)
        df_nx_graph.add_edge(starting_station, ending_station, tube_line=[], weight=average_time_taken, color=color)

# Include all tube lines that exist in each station
for line in range(df.shape[0]):
    starting_station = df.iloc[line,0]
    ending_station = df.iloc[line,1]
    tube_line = df.iloc[line,2]
    df_nx_graph.edges[starting_station, ending_station]['tube_line'].append(tube_line)

# Display graph
show_weighted_graph(df_nx_graph, 1500, 15, (50,50))
print(df_nx_graph)


# Implement DFS, BFS, UCS

# DFS
def my_recursive_dfs_implementation(graph, origin, destination, already_visited = [], count=1, reverse=False):
    # If reached destination, return list with the final place
    if origin == destination:
        return [origin], count+1

    next_already_visited = already_visited.copy()
    # Add current place to already_visited
    next_already_visited.append(origin)
    # Reverse list of neighbors if reverse is True
    neighbours = reversed(list(graph.neighbors(origin))) if reverse else graph.neighbors(origin)
    # Check all possible destinations from current node
    for next_node in neighbours:
        # Only explore nodes that have not been visited (No Loops nor going back)
        if next_node not in already_visited:
            # Go to first node possible
            result, count = my_recursive_dfs_implementation(graph, next_node, destination, next_already_visited, count, reverse)
            # If not dead end, means I found. Otherwise try next node
            if result != []:
                path = [origin] + result
                return path,count+1

    # If no node works, return empty string which means dead end
    return [], count+1


# BFS
def bfs_implementation(graph, origin, destination, counter = 0, reverse=False):
    # Add current place to already_visited
    next_already_visited = [origin]
    # List of existent paths (for now only origin)
    total_paths = [[origin]] 

    # Will perform exploration of all current paths
    while len(total_paths)!= 0: 
        new_total_paths = []
        # Check every single existing path for now
        for path in total_paths:
            # Last element in path
            last_element_in_path = path[-1]
            # Reverse list of neighbors if reverse is True
            nodes_found = list(reversed(list(graph.neighbors(last_element_in_path)))) if reverse else list(graph.neighbors(last_element_in_path))
            # Found destination
            if destination in nodes_found:
                # Result complete, will return this path with destination at end
                return path + [destination], counter+1

            # Otherwise, I'll need to explore the nodes connected to here...
            for node in nodes_found:
                # Only consider nodes that are not visited before (avoid loops and going back)
                if node not in next_already_visited:
                    counter += 1
                    # this node will be out of limits for next explorations
                    next_already_visited.append(node)
                    # Add to possible path for further exploration
                    new_total_paths = new_total_paths + [path + [node]]
            # Continue to explore these "new" paths, until I reach destination, or run out of possible valid paths
            total_paths = new_total_paths

    # If no more possible paths, means solution does not exist
    return [],-1


# UCS
def uniform_cost_search(nxobject, initial, goal, reverse=False):
    # Check to make sure initial is not equal to goal since we are checking children, or else return
    if initial == goal:
        return None  
    number_of_explored_nodes = 1
    # Implement frontier list for queue
    frontier = [{'label':initial, 'parent':None, 'time_cost':0}]  
    explored = {''}
    while frontier:
        node = frontier.pop(0) # pop from the left of the list
        # If current node is goal, return node, average time taken, and number of explored nodes
        if node['label'] == goal:
            return node, node['time_cost'], number_of_explored_nodes
        # If there are still nodes unexplored, continue to look at the neighbors
        if node['label'] not in explored:
            explored.add(node['label'])
            # Reverse list of neighbors if reverse is True
            neighbours = reversed(list(nxobject.neighbors(node['label']))) if reverse else nxobject.neighbors(node['label'])
            for child_label in neighbours:
                # Each child has weight (average time taken between one and next station)
                child_cost = nxobject.edges[node['label'], child_label]['weight'] + node['time_cost']
                child = {'label':child_label, 'parent':node, 'time_cost':child_cost}
                # If child has not been explored, add it into frontier and sort frontier based on the smallest to the highest cost       
                if child_label not in explored:
                    frontier = [child] + frontier
                    sort_frontier = sorted(frontier, key=lambda a : a.__getitem__('time_cost'))
                    frontier = sort_frontier
                    number_of_explored_nodes += 1
    return None


# Improve and implement current UCS cost function
# Extended UCS
def extended_uniform_cost_search(nxobject, initial, goal, reverse=False):
    # Check to make sure initial is not equal to goal since we are checking children, or else return
    if initial == goal: 
        return None  
    number_of_explored_nodes = 1
    # Implement frontier list for queue (for extended UCS, we include tube lines of stations)
    frontier = [{'label':initial, 'parent':None, 'time_cost':0, 'tube_line':None}]  
    explored = {''}
    while frontier:
        node = frontier.pop(0) # pop from the left of the list
        # If current node is goal, return node, average time taken, and number of explored nodes
        if node['label'] == goal:
            return node, node['time_cost'], number_of_explored_nodes
        if node['label'] not in explored:
            explored.add(node['label'])
            # Reverse list of neighbors if reverse is True
            neighbours = reversed(list(nxobject.neighbors(node['label']))) if reverse else nxobject.neighbors(node['label'])
            for child_label in neighbours:
                # Each child has weight (average time taken between starting and ending station)
                child_cost = nxobject.edges[node['label'], child_label]['weight'] + node['time_cost']
                # Each child has tube lines
                child_line = nxobject.edges[node['label'], child_label]['tube_line']
                for i in child_line:
                    # If the tube line of child node is the same as the parent node, means no lines are changed and cost would only be the average
                    # time taken to travel from one station to the next station
                    if i == node['tube_line']:
                        child = {'label':child_label, 'parent':node, 'time_cost':child_cost, 'tube_line':i}
                    # If the tube line of child node is different from the parent node, means the total cost needs to include the 
                    # time to change lines at one station
                    if i != node['tube_line']:
                        child = {'label':child_label, 'parent':node, 'time_cost':child_cost + 2, 'tube_line':i}
                    # If child has not been explored, add it into frontier and sort frontier based on the smallest to the highest cost
                    if child_label not in explored:
                        frontier = [child] + frontier
                        sort_frontier = sorted(frontier, key=lambda a : a.__getitem__('time_cost'))
                        frontier = sort_frontier
                        number_of_explored_nodes += 1
    return None


# Heuristic search

def heuristic(graph, start_zone, end_zone): # Calculates the admissible heuristic of a node
    initial_zone = int(graph.nodes[start_zone]['zone'])
    goal_zone = int(graph.nodes[end_zone]['zone'])
    
    return abs(initial_zone - goal_zone) # Return calculation of admissible heuristic (nearer zones means nearer to station)


# Heuristic BFS
def heuristic_bfs(graph, origin, goal):
    if origin == goal: # Check to make sure initial is not equal to goal since we are checking children, or else return
        return None
    admissible_heuristics = {} # Will save the values of h so i don't need to calculate multiple times for every node
    h = heuristic(graph, origin, goal)
    number_of_explored_nodes = 1
    admissible_heuristics[origin] = h
    visited_nodes = {} # This will contain the data of how to get to any node
    visited_nodes[origin] = (h, [origin]) # I add the data for the origin node: "Travel cost + heuristic", "Path to get there" and "Admissible Heuristic"

    paths_to_explore = PriorityQueue()
    paths_to_explore.put((h, [origin], 0)) # Add the origin node to paths to explore, also add cost without h
    # I add the total cost, as well as the path to get there (they will be sorted automatically)

    while not paths_to_explore.empty(): # While there are still paths to explore
        # Pop element with lower path cost in the queue
        _, path, total_cost = paths_to_explore.get()
        current_node = path[-1]
        if current_node == goal:
            return number_of_explored_nodes, visited_nodes[goal]
        neighbors = graph.neighbors(current_node) # I get all the neighbors of the current path
        
        for neighbor in neighbors:
            
            edge_data = graph.get_edge_data(path[-1], neighbor)
            if "weight" in edge_data:
                cost_to_neighbor = int(edge_data["weight"]) # If the graph has weights
            else:
                cost_to_neighbor = 1 # If the graph does not have weights I use 1

            if neighbor in admissible_heuristics:
                h = admissible_heuristics[neighbor]
            else:
                h = heuristic(graph, neighbor, goal)
                admissible_heuristics[neighbor] = h

            new_cost = total_cost + cost_to_neighbor
            new_cost_plus_h = new_cost + h
            # If this node was never explored, or the cost to get there is better than te previous ones
            if (neighbor not in visited_nodes) or (visited_nodes[neighbor][0]>new_cost_plus_h): 
                next_node = (new_cost_plus_h, path+[neighbor], new_cost)
                visited_nodes[neighbor] = next_node # Update the node with best value
                paths_to_explore.put(next_node) # Also will add it as a possible path to explore
                number_of_explored_nodes += 1
            
    return number_of_explored_nodes, visited_nodes[goal] # I will return the goal information, it will have the total cost without heuristic, the path, and the total cost with heuristic


# ### Results of heuristic search

# Results have the form of (path cost, path, path cost + heuristic cost)
hnum_visited1,hbfs_path1 = heuristic_bfs(df_nx_graph,"Euston","Victoria")
#print(hbfs_path1)
print("Total cost with heuristic: {}\nPath: {}\nTotal cost without heuristics: {}".format(hbfs_path1[0], hbfs_path1[1], hbfs_path1[2]))
hnum_visited2, hbfs_path2 = heuristic_bfs(df_nx_graph,"Canada Water","Stratford")
#print(hbfs_path2)
print("Total cost with heuristic: {}\nPath: {}\nTotal cost without heuristics: {}".format(hbfs_path2[0], hbfs_path2[1], hbfs_path2[2]))
hnum_visited3, hbfs_path3 = heuristic_bfs(df_nx_graph,"New Cross Gate","Stepney Green")
#print(hbfs_path3)
print("Total cost with heuristic: {}\nPath: {}\nTotal cost without heuristics: {}".format(hbfs_path3[0], hbfs_path3[1], hbfs_path3[2]))
hnum_visited4, hbfs_path4 = heuristic_bfs(df_nx_graph,"Ealing Broadway","South Kensington")
#print(hbfs_path4)
print("Total cost with heuristic: {}\nPath: {}\nTotal cost without heuristics: {}".format(hbfs_path4[0], hbfs_path4[1], hbfs_path4[2]))
hnum_visited5, hbfs_path5 = heuristic_bfs(df_nx_graph,"Baker Street","Wembley Park")
#print(hbfs_path5)
print("Total cost with heuristic: {}\nPath: {}\nTotal cost without heuristics: {}".format(hbfs_path5[0], hbfs_path5[1], hbfs_path5[2]))


# Functions for constructing paths, tube line paths, and path cost

def construct_path_from_root(node, root):
    # Construct path by looking from the root, parents of the each node starting from the final node
    path_from_root = [node['label']]
    while node['parent']:
        node = node['parent']
        path_from_root = [node['label']] + path_from_root
    return path_from_root

def construct_linepath_from_root(node, root):
    # Construct which tube line is used during the path, by looking at parents of the each node starting from the final node
    line_from_root = [node['tube_line']]
    while node['parent']:
        node = node['parent']
        linepath_from_root = [node['tube_line']] + line_from_root
    return linepath_from_root

def compute_path_cost(graph, path):
    cost = 0
    for index_station in range(len(path) - 1):
        cost += graph[path[index_station]][path[index_station + 1]]['weight']
    return cost


# ### Results of DFS, BFS, UCS, Extended UCS
# The following results are given: path in stations, cost in average time, number of node expanded.
# The following routes are considered:
# Euston to Victoria, Canada Water to Stratford, New Cross Gate to Stepney Green, Ealing Broadway to South Kensington, Baker Street to Wembley Park.

# Print results
# Display path in stations, cost in average time, number of node expanded
## Euston to Victoria
dfs_path1, number_visited_dfs1 = my_recursive_dfs_implementation(df_nx_graph, 'Euston', 'Victoria')
bfs_path1, number_visited_bfs1 = bfs_implementation(df_nx_graph, 'Euston', 'Victoria')
ucs_solution1, ucs_cost_path1, number_visited_ucs1 = uniform_cost_search(df_nx_graph, 'Euston', 'Victoria')
ucs_path1 = construct_path_from_root(ucs_solution1, 'Euston')
ucs_solution1ext, ucs_cost_path1ext, number_visited_ucs1ext = extended_uniform_cost_search(df_nx_graph, 'Euston', 'Victoria')
ucs_path1ext = construct_path_from_root(ucs_solution1ext, 'Euston')
ucs_linepath1ext = construct_linepath_from_root(ucs_solution1ext, 'Euston')

dfs_cost_path1 = compute_path_cost(df_nx_graph, dfs_path1)
bfs_cost_path1 = compute_path_cost(df_nx_graph, bfs_path1)


print('\nEuston to Victoria\n'+'='*10)
print("DFS Path: {}\nDFS Path Cost: {}\nDFS Number of visited nodes: {}".format(dfs_path1, dfs_cost_path1, number_visited_dfs1))
print("BFS Path: {}\nBFS Path Cost: {}\nBFS Number of visited nodes: {}".format(bfs_path1, bfs_cost_path1, number_visited_bfs1))
print("UCS Path: {}\nUCS Path Cost: {}\nUCS Number of visited nodes: {}".format(ucs_path1, ucs_cost_path1, number_visited_ucs1))
print("Extended UCS Path: {}\nExtended UCS Path Cost: {}\nExtended UCS Number of visited nodes: {}".format(ucs_path1ext, ucs_cost_path1ext, number_visited_ucs1ext))

## Canada Water to Stratford
dfs_path2, number_visited_dfs2 = my_recursive_dfs_implementation(df_nx_graph, 'Canada Water', 'Stratford', reverse=True)
bfs_path2, number_visited_bfs2 = bfs_implementation(df_nx_graph, 'Canada Water', 'Stratford')
ucs_solution2, ucs_cost_path2, number_visited_ucs2 = uniform_cost_search(df_nx_graph, 'Canada Water', 'Stratford')
ucs_path2 = construct_path_from_root(ucs_solution2, 'Canada Water')
ucs_solution2ext, ucs_cost_path2ext, number_visited_ucs2ext = extended_uniform_cost_search(df_nx_graph, 'Canada Water', 'Stratford')
ucs_path2ext = construct_path_from_root(ucs_solution2ext, 'Canada Water')
ucs_linepath2ext = construct_linepath_from_root(ucs_solution2ext, 'Canada Water')

dfs_cost_path2 = compute_path_cost(df_nx_graph, dfs_path2)
bfs_cost_path2 = compute_path_cost(df_nx_graph, bfs_path2)

print('\nCanada Water to Stratford\n'+'='*10)
print("DFS Path: {}\nDFS Path Cost: {}\nDFS Number of visited nodes: {}".format(dfs_path2, dfs_cost_path2, number_visited_dfs2))
print("BFS Path: {}\nBFS Path Cost: {}\nBFS Number of visited nodes: {}".format(bfs_path2, bfs_cost_path2, number_visited_bfs2))
print("UCS Path: {}\nUCS Path Cost: {}\nUCS Number of visited nodes: {}".format(ucs_path2, ucs_cost_path2, number_visited_ucs2))
print("Extended UCS Path: {}\nExtended UCS Path Cost: {}\nExtended UCS Number of visited nodes: {}".format(ucs_path2ext, ucs_cost_path2ext, number_visited_ucs2ext))

## New Cross Gate to Stepney Green
dfs_path3, number_visited_dfs3 = my_recursive_dfs_implementation(df_nx_graph, 'New Cross Gate', 'Stepney Green', reverse=True)
bfs_path3, number_visited_bfs3 = bfs_implementation(df_nx_graph, 'New Cross Gate', 'Stepney Green')
ucs_solution3, ucs_cost_path3, number_visited_ucs3 = uniform_cost_search(df_nx_graph, 'New Cross Gate', 'Stepney Green')
ucs_path3 = construct_path_from_root(ucs_solution3, 'New Cross Gate')
ucs_solution3ext, ucs_cost_path3ext, number_visited_ucs3ext = extended_uniform_cost_search(df_nx_graph, 'New Cross Gate', 'Stepney Green')
ucs_path3ext = construct_path_from_root(ucs_solution3ext, 'New Cross Gate')
ucs_linepath3ext  = construct_linepath_from_root(ucs_solution3ext, 'New Cross Gate')

dfs_cost_path3 = compute_path_cost(df_nx_graph, dfs_path3)
bfs_cost_path3 = compute_path_cost(df_nx_graph, bfs_path3)


print('\nCross Gate to Stepney Green\n'+'='*10)
print("DFS Path: {}\nDFS Path Cost: {}\nDFS Number of visited nodes: {}".format(dfs_path3, dfs_cost_path3, number_visited_dfs3))
print("BFS Path: {}\nBFS Path Cost: {}\nBFS Number of visited nodes: {}".format(bfs_path3, bfs_cost_path3, number_visited_bfs3))
print("UCS Path: {}\nUCS Path Cost: {}\nUCS Number of visited nodes: {}".format(ucs_path3, ucs_cost_path3, number_visited_ucs3))
print("Extended UCS Path: {}\nExtended UCS Path Cost: {}\nExtended UCS Number of explorations: {}".format(ucs_path3ext, ucs_cost_path3ext, number_visited_ucs3ext))


## Ealing Broadway to South Kensington
dfs_path4, number_visited_dfs4 = my_recursive_dfs_implementation(df_nx_graph, 'Ealing Broadway', 'South Kensington')
bfs_path4, number_visited_bfs4 = bfs_implementation(df_nx_graph, 'Ealing Broadway', 'South Kensington')
ucs_solution4, ucs_cost_path4, number_visited_ucs4 = uniform_cost_search(df_nx_graph, 'Ealing Broadway', 'South Kensington')
ucs_path4 = construct_path_from_root(ucs_solution4, 'Earling Broadway')
ucs_solution4ext, ucs_cost_path4ext, number_visited_ucs4ext = extended_uniform_cost_search(df_nx_graph, 'Ealing Broadway', 'South Kensington')
ucs_path4ext = construct_path_from_root(ucs_solution4ext, 'Earling Broadway')
ucs_linepath4ext = construct_linepath_from_root(ucs_solution4ext, 'Earling Broadway')


dfs_cost_path4 = compute_path_cost(df_nx_graph, dfs_path4)
bfs_cost_path4 = compute_path_cost(df_nx_graph, bfs_path4)


print('\nEaling Broadway to South Kensington\n'+'='*10)
print("DFS Path: {}\nDFS Path Cost: {}\nDFS Number of visited nodes: {}".format(dfs_path4, dfs_cost_path4, number_visited_dfs4))
print("BFS Path: {}\nBFS Path Cost: {}\nBFS Number of visited nodes: {}".format(bfs_path4, bfs_cost_path4, number_visited_bfs4))
print("UCS Path: {}\nUCS Path Cost: {}\nUCS Number of visited nodes: {}".format(ucs_path4, ucs_cost_path4, number_visited_ucs4))
print("Extended UCS Path: {}\nExtended UCS Path Cost: {}\nExtended UCS Number of visited nodes: {}".format(ucs_path4ext, ucs_cost_path4ext, number_visited_ucs4ext))

## Baker Street to Wembley Park
dfs_path5, number_visited_dfs5 = my_recursive_dfs_implementation(df_nx_graph, 'Baker Street', 'Wembley Park', reverse=True)
bfs_path5, number_visited_bfs5 = bfs_implementation(df_nx_graph, 'Baker Street', 'Wembley Park')
ucs_solution5, ucs_cost_path5, number_visited_ucs5 = uniform_cost_search(df_nx_graph, 'Baker Street', 'Wembley Park')
ucs_path5 = construct_path_from_root(ucs_solution5, 'Baker Street')
ucs_solution5ext, ucs_cost_path5ext, number_visited_ucs5ext = extended_uniform_cost_search(df_nx_graph, 'Baker Street', 'Wembley Park')
ucs_path5ext = construct_path_from_root(ucs_solution5ext, 'Baker Street')
ucs_linepath5ext = construct_linepath_from_root(ucs_solution5ext, 'Baker Street')

dfs_cost_path5 = compute_path_cost(df_nx_graph, dfs_path5)
bfs_cost_path5 = compute_path_cost(df_nx_graph, bfs_path5)

print('\nBaker Street to Wembley Park\n'+'='*10)
print("DFS Path: {}\nDFS Path Cost: {}\nDFS Number of visited nodes: {}".format(dfs_path5, dfs_cost_path5, number_visited_dfs5))
print("BFS Path: {}\nBFS Path Cost: {}\nBFS Number of visited nodes: {}".format(bfs_path5, bfs_cost_path5, number_visited_bfs5))
print("UCS Path: {}\nUCS Path Cost: {}\nUCS Number of visited nodes: {}".format(ucs_path5, ucs_cost_path5, number_visited_ucs5))
print("Extended UCS Path: {}\nExtended UCS Path Cost: {}\nExtended UCS Number of visited nodes: {}".format(ucs_path5ext, ucs_cost_path5ext, number_visited_ucs5ext))


# Comparison between UCS and Heuristic search
print('UCS\n'+'='*10)
print("UCS Path: {}\nUCS total cost: {}\nUCS Number of visited nodes: {}".format(ucs_path2, ucs_cost_path2, number_visited_ucs2))
print('\nHeuristic Search\n'+'='*10)
print("Total cost with heuristic: {}\nPath: {}\nTotal cost without heuristics: {}\nNumber of visited nodes: {}".format(hbfs_path2[0], hbfs_path2[1], hbfs_path2[2], hnum_visited2))

