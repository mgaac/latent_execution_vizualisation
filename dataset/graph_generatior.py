import networkx as nx
import mlx.core as mx
import numpy as np
import random
import pickle
from tqdm import tqdm
from collections import defaultdict

from enum import Enum

class task(Enum):
    PARALLEL_ALGORIHTM = 0
    SEQUENTIAL_ALGORITHM = 1

def binary_to_one_hot(binary_states):
    """Convert binary states (0/1) to one-hot vectors [reachable, not_reachable]"""
    # First, let's debug what we're getting
    num_timesteps = len(binary_states)
    num_nodes = len(binary_states[0]) if binary_states else 0
    
    # Create the array with explicit shape using Python lists first
    result_list = []
    
    for t, state_dict in enumerate(binary_states):
        timestep_data = []
        for node in sorted(state_dict.keys()):
            if state_dict[node] == 0:
                timestep_data.append([0, 1])  # not reachable
            else:
                timestep_data.append([1, 0])  # reachable
        result_list.append(timestep_data)
    
    # Convert to MLX array - this should preserve the 3D shape
    result = mx.array(result_list)
    return result

def detect_termination(states):
    """Detect when algorithm should terminate (no change between consecutive states)"""
    termination = []
    for i in range(len(states)):
        if i == len(states) - 1:  # Last step - algorithm terminates
            termination.append([0, 1])  # [continue, terminate]
        elif i < len(states) - 1:
            # Check if state changed from current to next
            current_state = states[i]
            next_state = states[i + 1]
            
            # Handle dictionary states (like distances) vs simple states
            if isinstance(current_state, dict) and isinstance(next_state, dict):
                states_equal = (current_state == next_state)
            else:
                states_equal = (current_state == next_state)
                
            if states_equal:
                termination.append([0, 1])  # [continue, terminate] - no change, should terminate
            else:
                termination.append([1, 0])  # [continue, terminate] - state changed, continue
        else:
            termination.append([1, 0])  # [continue, terminate] - default continue
    return mx.array(termination)

def generate_graphs(num_nodes):
    graphs = []

    # 1. Ladder graph
    G_ladder = nx.ladder_graph(num_nodes // 2)
    graphs.append(G_ladder)

    # 2. 2D grid graph
    side = int(np.sqrt(num_nodes))
    G_grid = nx.grid_2d_graph(side, side)
    G_grid = nx.convert_node_labels_to_integers(G_grid)
    graphs.append(G_grid)

    # 3. Random tree from Prüfer sequence
    G_tree = nx.random_spanning_tree(nx.complete_graph(num_nodes))
    graphs.append(G_tree)

    # 4. Erdős-Rényi graph
    p_er = min(np.log2(num_nodes) / num_nodes, 0.5)
    G_er = nx.erdos_renyi_graph(num_nodes, p_er)
    graphs.append(G_er)

    # 5. Barabási-Albert graph (attach 4 or 5 edges)
    m_ba = min(random.choice([4, 5]), num_nodes - 1)  # Ensure m < num_nodes
    if m_ba >= 1:  # Only create if valid
        G_ba = nx.barabasi_albert_graph(num_nodes, m_ba)
    else:
        # Fallback to a simple path graph for very small graphs
        G_ba = nx.path_graph(num_nodes)
    graphs.append(G_ba)

    # 6. 4-Community graph
    community_size = max(1, num_nodes // 4)  # Ensure at least 1 node per community
    if community_size * 4 <= num_nodes:
        communities = [nx.erdos_renyi_graph(community_size, 0.7) for _ in range(4)]
        G_4comm = nx.disjoint_union_all(communities)
        # Add inter-community edges
        nodes = list(G_4comm.nodes())
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if G_4comm.nodes[nodes[i]].get('block', i // community_size) != G_4comm.nodes[nodes[j]].get('block', j // community_size):
                    if random.random() < 0.01:
                        G_4comm.add_edge(nodes[i], nodes[j])
    else:
        # Fallback for very small graphs
        G_4comm = nx.complete_graph(num_nodes)
    graphs.append(G_4comm)

    # 7. 4-Caveman graph
    if community_size >= 1 and community_size * 4 <= num_nodes:
        G_caveman = nx.caveman_graph(4, community_size)
        # Remove intra-clique edges with probability 0.7
        to_remove = []
        for u, v in G_caveman.edges():
            if u // community_size == v // community_size and random.random() < 0.7:
                to_remove.append((u, v))
        G_caveman.remove_edges_from(to_remove)
        # Add shortcut edges
        num_shortcuts = int(0.025 * num_nodes)
        for _ in range(num_shortcuts):
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u // community_size != v // community_size:
                G_caveman.add_edge(u, v)
    else:
        # Fallback for very small graphs
        G_caveman = nx.cycle_graph(num_nodes)
    graphs.append(G_caveman)

    # Add self-loops and edge weights to all graphs
    for G in graphs:
        for n in G.nodes():
            G.add_edge(n, n)  # self-loop
        for u, v in G.edges():
            G[u][v]['weight'] = random.uniform(0.2, 1.0)

    return graphs

def edge_list_to_connection_matrix(edge_list):
    # Create source and target indices arrays
    sources = []
    targets = []
    weights = []
    
    for u, v, w in edge_list:
        sources.append(u)
        targets.append(v)
        weights.append(w)
    
    # Stack into a 3xN matrix where rows are [sources, targets, weights]
    return mx.array([sources, targets, weights])

def save_graphs(graphs, filename):
    processed_graphs = []
    
    for G in graphs:
        # Create edge list with weights
        edge_list = []
        for u, v in G.edges():
            weight = G[u][v]['weight']
            edge_list.append((u, v, weight))
            edge_list.append((v, u, weight))  # Add reverse edge
            
        connection_matrix = edge_list_to_connection_matrix(edge_list)
        # Generate targets for both tasks
        targets = {
            'parallel': generate_targets(edge_list, 0, task.PARALLEL_ALGORIHTM),
            'sequential': generate_targets(edge_list, 0, task.SEQUENTIAL_ALGORITHM)
        }
        
        processed_graphs.append({
            'connection_matrix': connection_matrix,
            'targets': targets
        })
    
    # Save the list of processed graphs
    with open(filename, 'wb') as f:
        pickle.dump(processed_graphs, f)

def load_graphs(filename):
    with open('utils/datasets/nega_custom/data/' + filename, 'rb') as f:
        return pickle.load(f)
    
def bfs_edge_list(edges, start):
    nodes = set([node for edge in edges for node in edge[:2]])
    state = {node: 1 if node == start else 0 for node in nodes}
    state_history = [state.copy()]
    for _ in range(len(nodes) - 1):
        new_state = state.copy()
        for u, v, _ in edges:
            if state[u] == 1:
                new_state[v] = 1
            if state[v] == 1:
                new_state[u] = 1
        state = new_state
        state_history.append(state.copy())
    return state_history

def calculate_stable_infinity(edges, start):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    lengths = nx.single_source_dijkstra_path_length(G, source=start, weight='weight')
    return max(lengths.values()) + 1

def bellman_ford_edge_list(edges, start):
    nodes = set([node for edge in edges for node in edge[:2]])
    INFINITY = calculate_stable_infinity(edges, start)
    distance = {node: 0 if node == start else INFINITY for node in nodes}
    predecessor = {node: node for node in nodes}
    history = [{'distance': distance.copy(), 'predecessor': predecessor.copy()}]
    for _ in range(len(nodes) - 1):
        for u, v, w in edges:
            if distance[u] != INFINITY and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                predecessor[v] = u
        history.append({'distance': distance.copy(), 'predecessor': predecessor.copy()})
    return history

def prim_edge_list(edges, start):
    adj_list = defaultdict(list)
    all_node_ids = set()
    for u, v, w in edges:
        adj_list[u].append((v, w))
        adj_list[v].append((u, w))
        all_node_ids.add(u)
        all_node_ids.add(v)

    if not edges and start is not None: # Handle case where start is given but no edges
        all_node_ids.add(start)
    
    if not all_node_ids: # Completely empty graph
        return [] 
    
    # Ensure start node is in the set of all nodes, even if isolated
    if start not in all_node_ids:
        # This case implies an invalid start node if other nodes/edges exist,
        # or an isolated start node to be added to the set.
        # For robustness, add it, assuming it's a valid scenario (e.g. single-node graph)
        all_node_ids.add(start)


    INFINITY = calculate_stable_infinity(edges, start) if edges else 0 # Avoid error if edges is empty

    state = {node: 0 for node in all_node_ids}
    distance = {node: INFINITY for node in all_node_ids}
    # internal_predecessor is used by the algorithm to find the MST
    internal_predecessor = {node: None for node in all_node_ids}

    if start not in all_node_ids : # Should not happen if logic above is correct
        # Fallback if start node was somehow missed (e.g. completely empty graph definition initially)
        # This indicates a potential issue with graph definition or start node.
        # However, to prevent crashes, initialize if necessary.
        # A robust solution might raise an error if start is invalid.
        # For now, we'll assume `start` is valid or becomes the only node.
        if not all_node_ids: all_node_ids.add(start) # Ensure start is in all_node_ids
        state[start] = 0
        distance[start] = INFINITY
        # internal_predecessor[start] will be set to start

    state[start] = 1
    distance[start] = 0
    internal_predecessor[start] = start  # MODIFICATION: Start node's predecessor is itself

    visited = {start}

    # Initialize distances from the start node using the adjacency list
    for v, w in adj_list.get(start, []):
        if w < distance[v]:
            distance[v] = w
            internal_predecessor[v] = start
    
    history = []

    # Helper function to prepare predecessor for history based on Equation (10)
    def get_history_predecessor_snapshot(current_state_dict, current_internal_preds_dict, graph_nodes_set, start_node_id):
        hist_preds_snapshot = {}
        for node_id in graph_nodes_set:
            if current_state_dict.get(node_id) == 1:  # Node is in MST (x_i^(t) = 1)
                if node_id == start_node_id:
                    hist_preds_snapshot[node_id] = start_node_id  # p_s = s
                else:
                    # For non-start nodes in MST, use their recorded internal predecessor
                    hist_preds_snapshot[node_id] = current_internal_preds_dict.get(node_id)
            else:  # Node is not in MST (x_i^(t) = 0)
                hist_preds_snapshot[node_id] = None  # Undefined / perp
        return hist_preds_snapshot

    # Record initial history (after start node is processed)
    initial_hist_preds = get_history_predecessor_snapshot(state, internal_predecessor, all_node_ids, start)
    history.append({'state': state.copy(), 'predecessor': initial_hist_preds})
    
    # Build MST
    while len(visited) < len(all_node_ids):
        # Select unvisited node with smallest distance to current MST
        reachable_unvisited_nodes = {
            node: dist for node, dist in distance.items() 
            if node not in visited and dist != INFINITY
        }
        
        if not reachable_unvisited_nodes:
            break  # No more reachable nodes (handles disconnected graphs)

        u = min(reachable_unvisited_nodes, key=reachable_unvisited_nodes.get)
        
        visited.add(u)
        state[u] = 1  # Mark node u as part of the MST

        # Update neighbor distances and internal_predecessors
        for v, w in adj_list.get(u, []):
            if v not in visited and w < distance[v]:
                distance[v] = w
                internal_predecessor[v] = u
        
        # Record history after adding this node u
        current_hist_preds = get_history_predecessor_snapshot(state, internal_predecessor, all_node_ids, start)
        history.append({'state': state.copy(), 'predecessor': current_hist_preds})
        
    return history

def generate_targets(graph, start, task_type):
    if task_type == task.PARALLEL_ALGORIHTM:
        bfs = bfs_edge_list(graph, start)
        bf = bellman_ford_edge_list(graph, start)

        sorted_nodes = sorted(bf[0]['distance'].keys()) if bf else []
        bf_dist = mx.array([[h['distance'][n] for n in sorted_nodes] for h in bf])
        bf_pred = mx.array([[h['predecessor'][n] for n in sorted_nodes] for h in bf])
        
        # Convert BFS state to one-hot and add termination targets
        bfs_state_one_hot = binary_to_one_hot(bfs)
        bfs_termination = detect_termination(bfs)
        
        # For Bellman-Ford, we also need to detect convergence for termination
        bf_states = [h['distance'] for h in bf]
        bf_termination = detect_termination(bf_states)

        # Debug print to see what shape we get

        return {
            'bfs_state': bfs_state_one_hot,
            'bfs_termination': bfs_termination,
            'bf_distance': bf_dist,
            'bf_predecessor': bf_pred,
            'bf_termination': bf_termination,
        }

    elif task_type == task.SEQUENTIAL_ALGORITHM:
        prim = prim_edge_list(graph, start)
        
        # Extract state history for one-hot conversion
        prim_states = [h['state'] for h in prim]
        prim_state_one_hot = binary_to_one_hot(prim_states)
        prim_termination = detect_termination(prim_states)
        
        # Debug print to see what shape we get
        
        # Handle potential None values in predecessor before converting to mx.array
        prim_predecessor_list = []
        for h in prim:
            current_preds = []
            # Ensure consistent node ordering if nodes dictionary is not sorted by default
            sorted_nodes = sorted(h['predecessor'].keys())
            for node_key in sorted_nodes:
                p_val = h['predecessor'][node_key]
                current_preds.append(-1 if p_val is None else p_val)
            prim_predecessor_list.append(current_preds)
        prim_predecessor = mx.array(prim_predecessor_list)

        return {
            'prim_state': prim_state_one_hot,
            'prim_termination': prim_termination,
            'prim_predecessor': prim_predecessor,
        }
    
if __name__ == "__main__":
    # Generate graphs with progress bars
    train_graphs_nested = [generate_graphs(20) for _ in tqdm(range(100), desc="Generating train graphs", leave=False)]
    val_graphs_nested = [generate_graphs(20) for _ in tqdm(range(5), desc="Generating validation graphs", leave=False)]
    test_graphs_20_nested = [generate_graphs(20) for _ in tqdm(range(5), desc="Generating test graphs (20 nodes)", leave=False)]
    test_graphs_50_nested = [generate_graphs(50) for _ in tqdm(range(5), desc="Generating test graphs (50 nodes)", leave=False)]
    test_graphs_100_nested = [generate_graphs(100) for _ in tqdm(range(5), desc="Generating test graphs (100 nodes)", leave=False)]

    # Flatten the lists
    train_graphs = [graph for graphs in train_graphs_nested for graph in graphs]
    val_graphs = [graph for graphs in val_graphs_nested for graph in graphs]
    test_graphs_20 = [graph for graphs in test_graphs_20_nested for graph in graphs]
    test_graphs_50 = [graph for graphs in test_graphs_50_nested for graph in graphs]
    test_graphs_100 = [graph for graphs in test_graphs_100_nested for graph in graphs]

    print("Saving graphs...\n")

    save_graphs(train_graphs, 'train_graphs.pkl')
    save_graphs(val_graphs, 'val_graphs.pkl')
    save_graphs(test_graphs_20, 'test_graphs_20.pkl')
    save_graphs(test_graphs_50, 'test_graphs_50.pkl')
    save_graphs(test_graphs_100, 'test_graphs_100.pkl')