import numpy as np
from Q3a import DecisionTree

class BaggedDecisionTrees:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=20,n_features=None,seed_state=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []
        self.n_features=n_features
        self.feature_subsets=[]
        self.OOBerror = None
        self.seed_state=seed_state
        # if self.seed_state is not None:
        #     np.random.seed(self.seed_state)
        # else: 
        #     np.random.seed(10)    
            

    def fit(self, X, y):
        self.feature_subsets=[]

        n_samples = X.shape[0]
        n_total_features=X.shape[1]

        oob_predictions = np.zeros((n_samples,))
        oob_counts = np.zeros((n_samples,))

        for _ in range(self.n_trees):

            #bootstrap 
            features_index = np.random.choice(n_samples,n_samples,replace=True)
            X_sample, y_sample = X[features_index], y[features_index]

            #select random subsets
            if self.n_features is not None: 
                chosen_feature = np.random.choice(n_total_features,self.n_features,replace=False)
                X_Subset = X_sample[:,chosen_feature]
            else:
                chosen_feature = np.arange(n_total_features)
                X_Subset=X_sample    


            tree = DecisionTree( max_depth=self.max_depth,min_samplescurrentsplitting=self.min_samples_split)
            tree.fit(X_Subset, y_sample)
            self.trees.append(tree)
            self.feature_subsets.append(chosen_feature)


            #OOB calc 
            oob_values = ~np.isin(np.arange(n_samples),features_index)
            oob_indices = np.where(oob_values)[0]

            if(len(oob_indices)>0):
                X_Oob = X[oob_indices][:,chosen_feature]
                oob_preds = tree.predict(X_Oob)

                oob_predictions[oob_indices] += oob_preds
                oob_counts[oob_indices] += 1

        valid_oob_mask = oob_counts > 0

        if np.any(valid_oob_mask):
            y_oob = y[valid_oob_mask]
            pred_oob = (oob_predictions[valid_oob_mask] / oob_counts[valid_oob_mask]) >= 0.5
            self.OOBerror = np.mean(pred_oob != y_oob)
        else:
            self.OOBerror = None 


    def predict(self,X):
        all_predictions=[]
        for tree,feature_index in zip(self.trees,self.feature_subsets):
            X_subset = X[:,feature_index]       
            preds = tree.predict(X_subset) 
            all_predictions.append(preds)
        return np.array(all_predictions).mean(axis=0) >= 0.5    

    # def predict(self, X):
        
    #     predictions = np.array([tree.predict(X) for tree in self.trees])
    #     # Majority vote for classification
    #     majority_vote = [np.bincount(pred).argmax() for pred in predictions.T]
    #     return np.array(majority_vote)
