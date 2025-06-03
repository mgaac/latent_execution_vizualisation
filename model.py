import mlx.nn as nn
import mlx.core as mx
from enum import Enum

class aggregation_fn(Enum):
    SUM = 1
    AVG = 2
    MIN = 4
    MAX = 5

class mp_layer(nn.Module):
    def __init__(self, embedding_dim: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum):
        super().__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.embedding_dim = embedding_dim

        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_fn = aggregation_fn

        self.source_message_fn = mx.random.normal([1, embedding_dim, embedding_dim])
        self.target_message_fn = mx.random.normal([1, embedding_dim, embedding_dim])

        self.update_fn = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def __call__(self, connection_matrix, node_embeddings):

        num_nodes = node_embeddings.shape[0]
        
        mask = mx.random.bernoulli(self.dropout_prob, connection_matrix.shape)
        connection_matrix = connection_matrix * mask

        edge_weights = mx.expand_dims(connection_matrix[2], axis=0)
        edge_weights = mx.expand_dims(edge_weights, axis=-1)

        connection_matrix = connection_matrix[:2].astype(mx.int32)

        node_embeddings = self.dropout(node_embeddings)

        source_idx = connection_matrix[self.source_idx]
        target_idx = connection_matrix[self.target_idx]

        source_embeddings = node_embeddings @ self.source_message_fn
        target_embeddings = node_embeddings @ self.target_message_fn

        filtered_source_embeddings = mx.take(source_embeddings, source_idx, axis=1)
        filtered_target_embeddings = mx.take(target_embeddings, target_idx, axis=1)

        message = filtered_source_embeddings + filtered_target_embeddings

        message = message * edge_weights

        message = self.relu(message)

        agg_message = mx.zeros([num_nodes, self.embedding_dim])

        if (self.aggregation_fn == aggregation_fn.SUM):
            agg_message = agg_message.at[target_idx].add(message)

        elif (self.aggregation_fn == aggregation_fn.AVG):
            agg_message = agg_message.at[target_idx].add(message)
            denominator = mx.zeros([num_nodes, 1]).at[target_idx].add(1)
            agg_message = agg_message /  mx.maximum(denominator, 1e-6)

        elif (self.aggregation_fn == aggregation_fn.MAX):
            agg_message = agg_message.at[target_idx].maximum(message)

        elif (self.aggregation_fn == aggregation_fn.MIN):
            agg_message = agg_message.at[target_idx].minimum(message)

        agg_message = self.dropout(agg_message)
        new_node_embeddings = self.update_fn(agg_message)
        new_node_embeddings = self.relu(new_node_embeddings)

        if (self.skip_connections):
            new_node_embeddings = new_node_embeddings + node_embeddings

        return new_node_embeddings

class mpnn(nn.Module):
    def __init__(self, embedding_dim: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int):
        super(mpnn, self).__init__()

        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_function_fn = aggregation_fn

        self.mp_layer = [
            mp_layer(embedding_dim, dropout_prob, skip_connections, aggregation_fn)
            for _ in range(num_mp_layers)
        ]

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        assert node_embeddings.shape[1] == self.embedding_dim, f'Incorrect node embedding size. Expected {self.embedding_dim}, got {node_embeddings.shape[1]}'

        for mp_layer in self.mp_layer:
            node_embeddings = mp_layer(connection_matrix, node_embeddings)

        return node_embeddings
    
class decoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super(decoder, self).__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.embedding_dim = embedding_dim
        self.predesecor_prob = nn.Linear(2 * embedding_dim, 1)

        self.bfs_state_outputs = nn.Linear(embedding_dim, 2)
        self.bfs_distance_outputs = nn.Linear(embedding_dim, 1)

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        num_nodes = node_embeddings.shape[0]

        connection_matrix = connection_matrix[:2].astype(mx.int32)

        source_idx = connection_matrix[self.source_idx]
        target_idx = connection_matrix[self.target_idx]

        source_embeddings = mx.take(node_embeddings, source_idx, axis=0)
        target_embeddings = mx.take(node_embeddings, target_idx, axis=0)

        concatenated_embeddings = mx.concat([source_embeddings, target_embeddings], axis=1)

        edge_scores = self.predesecor_prob(concatenated_embeddings)
        # edge_scores = (edge_scores - edge_scores.max()).exp()

        # softmax_denominator = mx.zeros([num_nodes, 1])

        # softmax_denominator = softmax_denominator.at[target_idx].add(edge_scores)
        # softmax_denominator = mx.take(softmax_denominator, target_idx, axis=0)

        # edge_prob = edge_scores / (softmax_denominator + 1e-16)
        edge_prob = mx.take(edge_scores, source_idx, axis=0)

        predesecor_predictions = mx.zeros([num_nodes, num_nodes])

        predesecor_predictions = predesecor_predictions.at[target_idx, source_idx].add(edge_prob.squeeze())

        bfs_state_predictions = self.bfs_state_outputs(node_embeddings)
        bf_distance_predictions = nn.relu(self.bfs_distance_outputs(node_embeddings))

        return bfs_state_predictions, bf_distance_predictions, predesecor_predictions
    
class nge(nn.Module):
    def __init__(self, embedding_dim: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int):
        super(nge, self).__init__()

        self.parallel_encoder = nn.Linear(2, embedding_dim)

        self.parallel_decoder = decoder(embedding_dim)

        self.parallel_termination_node = nn.Linear(embedding_dim, 2, bias=False)
        self.parallel_termination_global = nn.Linear(embedding_dim, 2, bias=False)
        self.parallel_termination_bias = mx.random.normal([2])
    
        self.processor = mpnn(embedding_dim, dropout_prob, skip_connections, aggregation_fn, num_mp_layers)

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        node_embeddings = self.parallel_encoder(node_embeddings)
        new_node_embeddings = self.processor((node_embeddings, connection_matrix))
        output = self.parallel_decoder((new_node_embeddings, connection_matrix))

        avg_node_embeddings = mx.mean(new_node_embeddings, axis=0)

        termination_prob = self.parallel_termination_node(new_node_embeddings) + self.parallel_termination_global(avg_node_embeddings) + self.parallel_termination_bias
        termination_prob = mx.mean(termination_prob, axis=0)

        return output, termination_prob