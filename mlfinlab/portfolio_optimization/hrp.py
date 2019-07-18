import pandas as pd


class HierarchicalRiskParity:

    def __init__(self):
        return

    def _tree_clustering(self):
        return

    def _quasi_diagnolization(self):
        return

    def _recursive_bisection(self):
        return

    def allocate(self, X):
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X)

        cov, corr = X.cov(), X.corr()