import matplotlib.pyplot as plt


# Removing the branch variable messes up the whole plot even though it is not used
# Plotting the tree using matplotlib

def plot_tree(node, feature_names, depth=0, position=(0, 0), parent_name=None, branch=0, s=20, linewidth=2,max_depth = 3):
    if node != None:
        # If node data is not None, then it is a leaf node and we plot the class label
        if node.data != None:
            plt.text(position[0], position[1], str(node.data), fontsize=12,
                     ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
        # If node data is None, then it is a decision node and we plot the feature name and the threshold
        else:
            feature_name = feature_names[node.feature]
            plt.text(position[0], position[1], f'{feature_name}\n<= {node.value}', fontsize=12,
                     ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
        # If parent_name is not None, then we plot the edge connecting the parent node to the current node
        if parent_name != None:
            plt.plot([parent_name[0], position[0]], [parent_name[1], position[1]], linewidth=linewidth, color='black')

        next_depth = depth + 1

        # If left and right child nodes are not None then we call plot_tree recursively on both the child nodes
        if node.left != None:
            next_position_left = (position[0] - 2 ** (max_depth - next_depth), position[1] - 2)
            plot_tree(node.left, feature_names, next_depth, next_position_left, position, 0, s, linewidth)

        if node.right != None:
            next_position_right = (position[0] + 2 ** (max_depth - next_depth), position[1] - 2)
            plot_tree(node.right, feature_names, next_depth, next_position_right, position, 1, s, linewidth)

