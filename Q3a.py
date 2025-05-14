# data 
import numpy as np
# Age Income Student Credit Rating Buy Computer
# 25 High No Fair No
# 30 High No Excellent No
# 35 Medium No Fair Yes
# 40 Low No Fair Yes
# 45 Low Yes Fair Yes
# 50 Low Yes Excellent No
# 55 Medium Yes Excellent Yes
# 60 High No Fair No

class TreeNode:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
        
    def leaf_node(self):
        return self.value    
    
class DecisionTree:
    def __init__(self, min_samplescurrentsplitting=2, max_depth=3, n_features=None):
        self.min_samplescurrentsplitting=min_samplescurrentsplitting
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None
        self.feature_types = None 

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.feature_types = [self._is_categorical(X[:, i]) for i in range(X.shape[1])]
        self.root = self._grow_tree(X, y)

    def _is_categorical(self, column):
       
        if isinstance(column[0], str):
            return True
        unique_values = np.unique(column)
        if len(unique_values) < 0.2 * len(column) and len(unique_values) < 10:
            return True
        return False
    

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # print(f"Depth {depth}: {n_samples} samples, {n_labels} unique labels")
        # print("X:\n", X)
        # print("y:", y)
        # print("-" * 50)

        # check the stopping criteria
        if (depth>=self.max_depth or n_samples<self.min_samplescurrentsplitting):
            leaf_value = self.current_comm_label(y)
            # print(f"Stopping Condition: depth={depth}, max_depth={self.max_depth}, n_labels={n_labels}, n_samples={n_samples}")
            return TreeNode(value=leaf_value)

        # feat_idxs = range(n_feats)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self.bestcurrentsplitting(X, y, feat_idxs)
        if best_feature is None:
            leaf_value = self.current_comm_label(y)
            return TreeNode(value=leaf_value)
        

        # if depth <= 5:
            # print(f"Level {depth} Split: Feature = {best_feature}, Threshold = {best_thresh}")
            
        
        # create child nodes
        left_idxs, right_idxs = self.currentsplitting(X[:, best_feature], best_thresh,self.feature_types[best_feature])
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        # print(best_thresh)
        # print(best_feature)
        return TreeNode(feature=best_feature,threshold= best_thresh, left=left, right=right)

    def current_comm_label(self, y):
    
        if len(y) == 0:
            return None
        counts = np.bincount(y)
        return np.argmax(counts)

    def bestcurrentsplitting(self,X,Y,features):
        best_red=-1
        split_index,split_threshold=None,None 
        for feat in features:
            X_col=X[:,feat]     
            thresholds=np.unique(X_col)

            for threshs in thresholds:
                #calc the impurity --> 1-sum p^2
                reduction = self.gini_impurity(Y,X_col,threshs,self.feature_types[feat])
                if reduction>best_red:
                    best_red=reduction
                    split_index=feat
                    split_threshold=threshs

        return (split_index,split_threshold) if best_red>0 else (None,None)
    
    # def bestcurrentsplitting(self, X, Y, features):
    #     best_red = -1
    #     split_index, split_threshold = None, None 

    #     for feat in features:
    #         X_col = X[:, feat]     
    #         thresholds = np.unique(X_col)

    #         for threshs in thresholds:
    #             reduction = self.gini_impurity(Y, X_col, threshs, self.feature_types[feat])

    #             # Changed condition to accept any improvement
    #             if reduction >= best_red or best_red == -1:
    #                 best_red = reduction
    #                 split_index = feat
    #                 split_threshold = threshs

    #     # Return best found split regardless of reduction amount
    #     return (split_index, split_threshold) if split_index is not None else (None, None)
    
    def gini_impurity(self,Y,Column, threshold,is_categorical):
        #parent gininindex 
        parent_gini=self.gini(Y)   

        left_feats,right_feats = self.currentsplitting(Column,threshold,is_categorical) 

        if len(left_feats) == 0 or len(right_feats) == 0:
            return 0   
        
        #weightedgini
        n=len(Y)
        num_left,num_right=len(left_feats),len(right_feats)
        gini_left,gini_right=self.gini(Y[left_feats]),self.gini(Y[right_feats])

        weighted_gini = (num_left / n) * gini_left + (num_right / n) * gini_right
        gini_reduction = parent_gini - weighted_gini
        
        return gini_reduction


    def gini(self, Y):
        if len(Y) == 0:
            return 0
        counts = np.bincount(Y)
        probs = counts / len(Y)
        return 1 - np.sum(probs**2)
    
    def currentsplitting(self, Col, thr,is_categorical):
        if is_categorical:
            # Split categorical data by equality
            left_feat = np.argwhere(Col == thr).flatten()
            right_feat = np.argwhere(Col != thr).flatten()
        else:
            # Split numerical data as usual
            left_feat = np.argwhere(Col <= thr).flatten()
            right_feat = np.argwhere(Col > thr).flatten()

        return left_feat, right_feat

        
    def gini(self,Y):
        tot_label=np.bincount(Y)
        ps=tot_label/len(Y)

        #gini impurity is 1-sum of pi^2
        gini_imp = 1 - np.sum([p**2 for p in ps ])

        return gini_imp
    
    def predict(self,X):
        return  np.array([self.traverse(x,self.root) for x in X])

    def traverse(self, x, node):
   
    # print(f"Node: {node.feature}, Threshold: {node.threshold}, Value: {node.value}")

        if node.value is not None:
            # print(f"Reached leaf: {node.value}")
            return node.value

        feature_val = x[node.feature]

        if self.feature_types[node.feature]:
            # Categorical feature
            if feature_val == node.threshold:
                # print(f"Feature {node.feature} ({feature_val}) == Threshold ({node.threshold}) ->Left")
                return self.traverse(x, node.left)
            else:
                # print(f"Feature {node.feature} ({feature_val}) != Threshold ({node.threshold}) -> Right")
                return self.traverse(x, node.right)
        else:
            # Numerical feature
            if feature_val <= node.threshold:
                # print(f"Feature {node.feature} ({feature_val}) <= Threshold ({node.threshold}) -> Left")
                return self.traverse(x, node.left)
            else:
                # print(f"Feature {node.feature} ({feature_val}) > Threshold ({node.threshold}) ->Right")
                return self.traverse(x, node.right)
            

    # def traverse(self, x, node):
    #     print(f"Node: {node.feature}, Threshold: {node.threshold}, Value: {node.value}")

    #     if node.value is not None:
    #         print(f"Reached leaf: {node.value}")
    #         return node.value

    #     feature_val = x[node.feature]

    #     if self.feature_types[node.feature]:
    #         # Categorical feature
    #         if feature_val == node.threshold:
    #             print(f"Feature {node.feature} ({feature_val}) == Threshold ({node.threshold}) ->Left")
    #             return self.traverse(x, node.left)
    #         else:
    #             print(f"Feature {node.feature} ({feature_val}) != Threshold ({node.threshold}) -> Right")
    #             return self.traverse(x, node.right)
    #     else:
    #         # Numerical feature
    #         if feature_val <= node.threshold:
    #             print(f"Feature {node.feature} ({feature_val}) <= Threshold ({node.threshold}) -> Left")
    #             return self.traverse(x, node.left)
    #         else:
    #             print(f"Feature {node.feature} ({feature_val}) > Threshold ({node.threshold}) ->Right")
    #             return self.traverse(x, node.right)
