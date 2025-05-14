from Q3a import DecisionTree
from Q4_2023566 import BaggedDecisionTrees
import numpy as np 

# clf=DecisionTree()
# bagged_clf = BaggedDecisionTrees(n_trees=10, max_depth=5)

# data = np.array([
#     [25, 2, 0, 0, 0],  # 25, High, No, Fair, No
#     [30, 2, 0, 1, 0],  # 30, High, No, Excellent, No
#     [35, 1, 0, 0, 1],  # 35, Medium, No, Fair, Yes
#     [40, 0, 0, 0, 1],  # 40, Low, No, Fair, Yes
#     [45, 0, 1, 0, 1],  # 45, Low, Yes, Fair, Yes
#     [50, 0, 1, 1, 0],  # 50, Low, Yes, Excellent, No
#     [55, 1, 1, 1, 1],  # 55, Medium, Yes, Excellent, Yes
#     [60, 2, 0, 0, 0],  # 60, High, No, Fair, No
# ])

data = np.array([
    [25, "High", "No", "Fair", "No"],
    [30, "High", "No", "Excellent", "No"],
    [35, "Medium", "No", "Fair", "Yes"],
    [40, "Low", "No", "Fair", "Yes"],
    [45, "Low", "Yes", "Fair", "Yes"],
    [50, "Low", "Yes", "Excellent", "No"],
    [55, "Medium", "Yes", "Excellent", "Yes"],
    [60, "High", "No", "Fair", "No"],
])

X = data[:, :-1]  
y = data[:, -1]


y = np.array([1 if label == "Yes" else 0 for label in y])

# Initialize all the trees
normal_decision=DecisionTree()
bagged_clf = BaggedDecisionTrees(n_trees=10, max_depth=3)
random_forest=BaggedDecisionTrees(n_trees=10,max_depth=3,n_features=2)


normal_decision.fit(X,y)
bagged_clf.fit(X, y)
random_forest.fit(X,y)



# Predict
test_case = np.array([[42, "Low", "No", "Excellent"]], dtype=object)


normal=normal_decision.predict(test_case)
prediction_bagged = bagged_clf.predict(test_case)
prediction_randomforrest=random_forest.predict(test_case)


print("Bagged Prediction:", "Yes" if prediction_bagged[0] == 1 else "No")
print("Normal Prediction:", "Yes" if normal[0] == 1 else "No")
print("Random Forest Prediction:", "Yes" if prediction_randomforrest[0] == 1 else "No")
print("Bagged OOB error",bagged_clf.OOBerror)
print("Random forest OOB error",random_forest.OOBerror)
