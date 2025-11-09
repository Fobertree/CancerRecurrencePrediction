import torch

class HypergraphBuilder():
    def __init__(self, df):
        self.df = df
    
    def get_edge_index(self):
        '''
        This assumes everything is one hot-encoded (either via binning or directly)
        '''
        node_idx = []
        edge_idx = []

        n_nodes, n_hyperedges = self.df.shape

        for i in range(n_nodes):
            for j in range(n_hyperedges):
                if self.df.iloc[i, j] != 0:
                    node_idx.append(i)
                    edge_idx.append(j)

        edge_index = torch.tensor([node_idx, edge_idx], dtype=torch.long)
        return edge_index

    def get_data(self, add_x=True):
        """
        Optionally return a torch_geometric.data.Data object.
        """
        from torch_geometric.data import Data

        edge_index = self.get_edge_index()
        x = torch.tensor(self.df.values, dtype=torch.float) if add_x else None

        return Data(x=x, edge_index=edge_index)

if __name__ == "__main__":
    pass