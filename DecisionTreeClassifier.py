import math

class DecisionTreeClassifier:
    """
    ID3 Decision Tree Classifier

    Author: Ethan Cramer
    """
    class Node:
        """ 
        Node Subclass to handle each node of the decision tree
        
        - value: the value of the node
        - next: the node object next in line
        - children: the children nodes of this node

        """
        def __init__(self):
            self.value = None
            self.next = None
            self.children = None

    def __init__(self):
        self.data = None # DataFrame containing the training feature columns
        self.labels = None # DataFrame containing the training labels columns
        self.feature_names = None # Column names
        self.node = None # Starting node of the decision tree
        self.totalEntropy = None # Total entropy calculation

    def fit(self, data, labels):
        """
        Fit a tree to training data
        
        Input:
        - data [DataFrame]: training feature columns
        - labels [DataFrame]: training label columns
        
        """
        self.data = data
        self.labels = labels
        self.feature_names = list(data.columns)
        self.totalEntropy = self.calc_entropy([row for row in range(len(self.data))])
        
        rows = [row for row in range(len(self.data))] # samples
        columns = [column for column in range(len(self.feature_names))] # features
        self.node = self.generate_tree(rows, columns, self.node) # recursively generate decision tree

    def predict(self, instance, *root_node):
        """
        Predict a label given the feature values

        Input:
        - instance [Series]: single row of feature values
        - {optional} root_node [Node Object]: new starting point

        Output:
        - root_node.value [str]: predicted label of instance

        """

        if not root_node:
            root_node = self.node # Set the root_node at the start of the tree
        else:
            root_node = root_node[0]  # Set the root_node to the new starting point

        if not root_node.children and not root_node.next: # Once a leaf node has been found
            return root_node.value # Prediction
        else:
            feature_value = instance[root_node.value]
            if root_node.children is not None:
                for i in range(len(root_node.children)):
                    # If a feature value in the current instance is found in a child, set the new root as that child
                    if feature_value == root_node.children[i].value:
                        return self.predict(instance, root_node.children[i].next) # Recursively parse through the tree
            elif root_node.next is not None:
                # If no children contained a match, then go on to the next node
                return self.predict(instance, root_node.next) # Recursively parse through the tree
            else:
                return None

    def evaluate(self, test_data, test_labels):
        """
        Evaluate the accuracy of the decision tree

        Input:
        - test_data [DataFrame]: test feature columns
        - test_labels [DataFrame]: test label columns

        Output:
        - accuracy [float]: the fraction of correct predictions

        """
        correct = 0
        wrong = 0

        for index, _ in test_data.iterrows(): 
            result = self.predict(test_data.iloc[index])

            if result == test_labels.iloc[index]:
                correct += 1
            else:
                wrong += 1

        accuracy = correct / (correct + wrong)

        return accuracy

    def calc_entropy(self, rows):
        """
        Calculate the entropy of a set of rows

        Input:
        - rows [int]: sample indexes

        Output:
        - entropy [float]: calculated entropy

        """
        # filter the labels
        labels = [self.labels.iloc[row] for row in rows]
        # get the label categories (e.g. 'unacc', 'acc', 'good', 'vgood')
        label_categories = list(set(self.labels))
        # get the count of each category
        label_category_count = [labels.count(category) for category in label_categories]
        # get the total count of rows
        total_count = len(rows)

        # Total Entropy = - p(good)*log2(p(good)) - p(vgood)*log2(p(vgood)) - ...
        entropy = sum([-(count / total_count) * math.log(count / total_count, 2) if count != 0 else 0 for count in label_category_count])

        return entropy

    def calc_info_gain(self, rows, column):
        """
        Calculate the information gain of a given feature

        Input:
        - rows [int]: sample indexes
        - column [int]: index of current feature

        Output:
        - info_gain [float]: informationg gain of current feature
        
        """
        # Row indexes of the current feature
        feature_rows = [self.data.iloc[row][column] for row in rows]
        # Categories of the current feature
        feature_categories = list(set(feature_rows)) 
        # Number of instances of each category
        feature_count = [feature_rows.count(row) for row in feature_categories]
        # Split up the rows by category so we can find the entropy of a single feature category at a time
        feature_category_row = [[rows[i] for i, x in enumerate(feature_rows) if x == y] for y in feature_categories]
        # Probability = Current Feature Count / Feature Rows
        feature_info = sum([count / len(rows) * self.calc_entropy(row) for count, row in zip(feature_count, feature_category_row)])
        # Info Gain = Total Entropy - Feature Info
        info_gain = self.totalEntropy - feature_info

        return info_gain

    def get_best_feature(self, rows, columns):
        """
        Find the feature that maximizes information gain

        Input:
        - rows [int]: sample indexes
        - columns [int]: feature indexes

        Output:
        - self.feature_names[best_feature_column] [str]: best feature name
        - best_feature_column [int]: best feature column index

        """
        # Calculate the information gain for each feature
        info_gain = [self.calc_info_gain(rows, column) for column in columns]
        best_feature_column = columns[info_gain.index(max(info_gain))]

        return self.feature_names[best_feature_column], best_feature_column

    def generate_tree(self, rows, columns, node):
        """
        Populate the tree with nodes

        Input:
        - rows [int]: sample indexes
        - columns [int]: feature indexes
        - node [Node object]: current node (will update recursively)
        
        """
        # Create then Node on the first iteration
        if not node:
            node = DecisionTreeClassifier.Node()

        # List of labels corresponding to rows
        labels_in_instance = [self.labels[row] for row in rows]

        # Check if we've reached a pure node
        if len(set(labels_in_instance)) == 1:
            node.value = set(labels_in_instance).pop()
            return node

        # Check if we've run out of features
        if len(columns) == 0:
            node.value = max(set(labels_in_instance), key = labels_in_instance.count)
            return node

        # Choose the feature that maximizes informationg gain to progress the tree
        best_feature, best_feature_column = self.get_best_feature(rows, columns)
        node.value = best_feature
        node.children = []

        # Create children nodes for each category of the best feature
        feature_categories = list(set([self.data.iloc[row][best_feature_column] for row in rows]))
            # Would this work? 
            # feature_categories = list(set(self.data[rows][best_feature_column]))
        
        for category in feature_categories:
            child = DecisionTreeClassifier.Node()
            child.value = category

            # Append child to the root node
            node.children.append(child)

            # Create the set of rows where the row matches the current category
            child_rows = [row for row in rows if self.data.iloc[row][best_feature_column] == category]

            # If there are no rows matching the current category, go to the next most probably category
            if not child_rows:
                child.next = max(set(labels_in_instance), key = labels_in_instance.count)
            # If there are rows of the current category, remove the current column and recurse
            else:
                if columns and best_feature_column in columns:
                    columns.pop(columns.index(best_feature_column))
                child.next = self.generate_tree(child_rows, columns, child.next)

        return node