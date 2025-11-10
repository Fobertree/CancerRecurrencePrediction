import torch
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

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
        print(self.df.values[0],  self.df.columns)
        input()
        x = torch.tensor(self.df.values, dtype=torch.int64) if add_x else None

        return Data(x=x, edge_index=edge_index)

if __name__ == "__main__":

    # replace this with your metadata path
    METADATA_PATH = "/Users/alexanderliu/EmoryCS/CancerRecurrencePrediction/new_metadata.csv"

    # DF LOAD + PREPROCESS
    def load_metadata_features():
        df = pd.read_csv(METADATA_PATH)
        # drop first to prevent multicollinearity
        df = pd.get_dummies(df, columns=["HistologicType"], drop_first=True)

        continuous_cols = ['Age', 'TumorSize']

        # Initialize KBinsDiscretizer for 4 bins using 'quantile' strategy and 'ordinal' encoding
        n_bins = 4
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

        # Apply discretization to the selected columns
        df_discretized_values = discretizer.fit_transform(df[continuous_cols])

        # Create a new DataFrame with the discretized columns
        df_discretized = pd.DataFrame(df_discretized_values, columns=[col + '_binned' for col in continuous_cols])

        # Combine with original non-discretized columns (e.g., categorical_col)
        df = pd.concat([df.drop(columns=continuous_cols), df_discretized], axis=1)
        return df

    df = load_metadata_features()
    hb = HypergraphBuilder(df)
    dataset = hb.get_data()
    print(dataset)