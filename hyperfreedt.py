"""Implementation of A Novel Hyperparameter-free Approach
to Decision Tree Construction that Avoids Overfitting by Design"""

import bz2
from dataclasses import dataclass
from typing import Self, Tuple

import numpy as np

# max number of splits to test for a feature
MAX_THRESHOLDS = 20

# small value to avoid division by zero
EPSILON = 1e-10


@dataclass
class Split:
    """Tree node split"""

    feature: int
    threshold: float


@dataclass
class TreeNode:
    """Node of a decision tree"""

    left_child: Self | None
    right_child: Self | None
    split: Split | None
    row_indices: np.ndarray
    forecast: np.ndarray


@dataclass
class Solution:
    """Best candidate node info"""

    node: TreeNode
    split: Split
    left_child: TreeNode
    right_child: TreeNode


def forecast(y: np.ndarray, class_count: int) -> np.ndarray:
    """turn class labels into class probabilities"""
    return np.bincount(y, minlength=class_count) / y.shape[0]


def get_best_split(x: np.ndarray, y: np.ndarray, class_count: int) -> Split | None:
    """Find best split (feature index and threshold) for the given data"""
    best_weighted_entropy = float("inf")
    best_split = None
    for feature_ix in range(x.shape[1]):
        feature_x = x[:, feature_ix]
        uniq_vals = np.sort(np.unique(feature_x))
        if len(uniq_vals) > MAX_THRESHOLDS:
            uniq_vals = np.unique(
                np.percentile(
                    feature_x,
                    [100 / MAX_THRESHOLDS * i for i in range(1, MAX_THRESHOLDS)],
                )
            )
        for threshold in uniq_vals[1:]:
            left_y = y[feature_x < threshold]
            right_y = y[feature_x >= threshold]

            left_p = np.bincount(left_y.astype(int), minlength=class_count) / len(
                left_y
            )
            right_p = np.bincount(right_y.astype(int), minlength=class_count) / len(
                right_y
            )
            left_p = left_p[left_p > 0]
            right_p = right_p[right_p > 0]
            left_entropy = -np.sum(left_p * np.log2(left_p)) if left_p.size else 0
            right_entropy = -np.sum(right_p * np.log2(right_p)) if right_p.size else 0

            left_weight = left_y.shape[0] / y.shape[0]
            right_weight = right_y.shape[0] / y.shape[0]
            weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
            if weighted_entropy < best_weighted_entropy:
                best_weighted_entropy = weighted_entropy
                best_split = Split(feature_ix, threshold)
    return best_split


def get_split_indices(
    x: np.ndarray, node: TreeNode, split: Split
) -> Tuple[np.ndarray, np.ndarray]:
    """Get row indices of left and right splits"""
    node_data = x[node.row_indices]
    left_ix = node.row_indices[node_data[:, split.feature] < split.threshold]
    right_ix = node.row_indices[node_data[:, split.feature] >= split.threshold]
    return left_ix, right_ix


def compressed_length_bz2(data: bytes) -> int:
    """
    Compresses a numpy array of 0s and 1s using bzip2 and returns the length of the compressed data.

    Parameters:
    arr (numpy.ndarray): The input array containing only 0s and 1s.

    Returns:
    int: Length of the compressed data in bytes.
    """
    compressed = bz2.compress(data)
    return len(compressed)


def predict(tree: TreeNode, x: np.ndarray) -> np.ndarray:
    """Get predicted class probabilities for data using a decision tree"""
    predictions = []
    for row in x:
        current_node = tree
        while current_node.split:
            if row[current_node.split.feature] < current_node.split.threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        predictions.append(current_node.forecast)
    return np.array(predictions)


def compute_inaccuracy(
    tree: TreeNode, x: np.ndarray, y: np.ndarray, use_original_incaccuracy: bool
) -> float:
    """Compute inaccuracy score for data using a decision tree"""
    class_probs = predict(tree, x)
    if class_probs.ndim == 1:
        prediction = np.digitize(class_probs, [0.5])
    else:
        prediction = np.argmax(class_probs, axis=1)
    errors = prediction != y

    if use_original_incaccuracy:
        # inaccuracy calculation from the research paper
        misclassified_y = y[errors]
        return compressed_length_bz2(misclassified_y.tobytes()) / compressed_length_bz2(
            y.tobytes()
        )
    else:
        # return error rate
        return np.sum(errors) / len(y)


def format_branches(tree: TreeNode, attributes: set, level: int = 1) -> str:
    """Return a pythonish string representation of a tree branch"""
    branches_str = ""
    if tree.split:
        attributes.add(f"X{tree.split.feature}")
        branches_str += (
            f"{'    '*level}if X{tree.split.feature} < {tree.split.threshold}:\n"
        )
        branches_str += format_branches(tree.left_child, attributes, level + 1)
        branches_str += f"{'    '*level}else:\n"
        branches_str += format_branches(tree.right_child, attributes, level + 1)
    else:
        branches_str += f"{'    '*level}return {np.argmax(tree.forecast)}\n"
    return branches_str


def format_tree(tree: TreeNode) -> str:
    """Return a pythonish string representation of a decision tree"""
    attributes = set()
    branches = format_branches(tree, attributes)
    return f"def tree{{{','.join(attributes)}}}:\n{branches}"


def compute_complexity(tree: TreeNode) -> float:
    """Compute complexity of a decision tree"""
    model_string = format_tree(tree)
    return compressed_length_bz2(model_string.encode("utf-8")) / len(model_string)


def compute_tree_cost(
    tree: TreeNode, x: np.ndarray, y: np.ndarray, use_original_incaccuracy
) -> float:
    """Compute cost of a decision tree given some data"""
    if tree.split is None:
        # Return a large cost for the root node so it always gets split
        return 1
    inaccuracy = compute_inaccuracy(tree, x, y, use_original_incaccuracy)
    complexity = compute_complexity(tree)
    # return harmonic mean of inaccuracy and complexity
    return 2 / (1 / (inaccuracy + EPSILON) + 1 / complexity)


def check_args(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make sure feature and target data meet requirements"""
    if not isinstance(x, np.ndarray):
        raise ValueError("parameter `x` should be a ndarray")
    if not isinstance(y, np.ndarray):
        raise ValueError("parameter `y` should be a ndarray")

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()

    if x.ndim != 2:
        raise ValueError("parameter `x` should be two dimensional")
    if y.ndim != 1:
        raise ValueError("parameter `y` should be one dimensional")

    if not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"parameter `x` must be a numeric array and not {x.dtype}")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"parameter `y` must be a numeric array and not {y.dtype}")

    y_int = y.astype(int)
    if not np.allclose(y, y_int):
        raise ValueError("parameter `y` should contain integer class IDs")

    return x, y_int


def build_tree(
    x: np.ndarray, y: np.ndarray, use_original_incaccuracy: bool = False
) -> TreeNode:
    """
    Builds a decision tree from the given training data.

    Args:
        x (np.ndarray): The feature matrix where each row represents a
            sample and each column represents a feature.
        y (np.ndarray): The target vector where each element corresponds
            to the class label of a sample.
        use_original_incaccuracy (bool, optional): Flag to determine whether to use
            the original inaccuracy metric from the research paper for computing tree cost.
            Defaults to False.

    Returns:
        TreeNode: The root node of the constructed decision tree.
    """
    x, y = check_args(x, y)
    class_count = np.amax(y) + 1
    fc = forecast(y, class_count)
    root = TreeNode(None, None, None, np.arange(x.shape[0]), fc)
    best_cost = compute_tree_cost(root, x, y, use_original_incaccuracy)
    candidates = [root]
    while candidates:
        best_solution = None
        candidates_to_remove = []
        for candidate in candidates:
            candidate_x = x[candidate.row_indices]
            candidate_y = y[candidate.row_indices]
            best_split = get_best_split(candidate_x, candidate_y, class_count)
            if not best_split:
                candidates_to_remove.append(id(candidate))
            else:
                left_ix, right_ix = get_split_indices(x, candidate, best_split)
                left_fc = forecast(y[left_ix], class_count)
                right_fc = forecast(y[right_ix], class_count)
                left_child = TreeNode(None, None, None, left_ix, left_fc)
                right_child = TreeNode(None, None, None, right_ix, right_fc)
                candidate.left_child = left_child
                candidate.right_child = right_child
                candidate.split = best_split
                tree_cost = compute_tree_cost(root, x, y, use_original_incaccuracy)
                candidate.left_child = None
                candidate.right_child = None
                candidate.split = None
                if tree_cost < best_cost:
                    best_cost = tree_cost
                    best_solution = Solution(
                        candidate, best_split, left_child, right_child
                    )
        candidates = [c for c in candidates if id(c) not in candidates_to_remove]
        if best_solution:
            best_solution.node.left_child = best_solution.left_child
            best_solution.node.right_child = best_solution.right_child
            best_solution.node.split = best_solution.split
            candidates = [c for c in candidates if c != best_solution.node]
            candidates.append(best_solution.node.left_child)
            candidates.append(best_solution.node.right_child)
        else:
            return root
    return root


def print_tree(tree: TreeNode | None, level: int = 0) -> None:
    """Print a representation of a decision tree's structure"""
    if tree is None:
        return
    if tree.split:
        print(
            f"{' '*level}{tree.split.feature} < {tree.split.threshold} = {tree.forecast}"
        )
    else:
        print(f"{' '*level} leaf = {tree.forecast}")
    print_tree(tree.left_child, level + 1)
    print_tree(tree.right_child, level + 1)


def count_leaves(tree: TreeNode) -> int:
    """Count the number of leaf nodes in a decision tree"""
    if tree.left_child is None:
        return 1
    return count_leaves(tree.left_child) + count_leaves(tree.right_child)


def test_build() -> None:
    """Test tree builder"""
    x = np.random.rand(1000, 5)
    y = np.sum(x, axis=1) + np.random.normal(0, 1, 1000)
    y = (y > np.mean(y)).astype(int)
    tree = build_tree(x, y)
    print_tree(tree)


if __name__ == "__main__":
    test_build()
