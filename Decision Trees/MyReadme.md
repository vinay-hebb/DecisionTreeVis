<style>
  ques { background-color: red; color: white; padding: 1px; border-radius: 2px; }
  note { background-color: yellow; color: black; padding: 1px; border-radius: 2px; }
  imp { background-color: lightgreen; color: black; padding: 1px; border-radius: 2px; }
  underline1 { text-decoration: underline; }
</style>
# A quick demonstration of WebApp

A GIF can communicate lot more

![WebApp Demo](Demo1.gif)

If you are well versed with the mathematical formulation of decision tree, you can skip to the [Webapp Section](#webapp) directly.

# Decision Tree Classifier: Mathematical Formulation (scikit-learn)

A **Decision Tree Classifier** is a supervised learning algorithm that recursively partitions the feature space to classify data points. The scikit-learn implementation uses the **CART** (Classification and Regression Trees) algorithm, which builds binary trees using the feature and threshold that yield the largest information gain (or impurity reduction).

# Background
Given
- $x_i \in \mathbb{R}^n,\quad i = 1, \ldots, l$
- $y \in \mathbb{R}$
<!-- - $y \in \mathbb{R}^l$<ques>Is it l or scalar?</ques> -->

Goal is to find an estimator which estimates $y$ reliably for a new example $x$. A way to estimate new example's label is by splitting the given region and labelling split regions as a particular label if we have sufficient confidence. Any new example is assigned that label if it falls in the given region. A computationally simplest way to split into regions is to use thresholds on particular features. To gain confidence of regions, we can split it into many regions recursively. This can be represented as tree with nodes. Each node has a split condition and set of examples associated with it. 

# Notations
- Consider a node $m$ and let feature be $j$ on which we make a split.
- Let us denote $\theta=(j, t_m)$ and $t_m$ is threshold which separates 2 regions
- $Q_m$ is data present at node $m$
- $Q_m^{left}(\theta)$ and $Q_m^{right}(\theta)$ are 2 regions split by node $m$. Mathematically, they can be written as below

$$
\begin{aligned}
Q_m^{left}(\theta)  & = \{(x, y) \mid x_j \leq t_m\} \\
Q_m^{right}(\theta) & = Q_m \setminus Q_m^{left}(\theta)
\end{aligned}
$$

We have setup the regions but we have not answered, when do we say we have sufficient "confidence" to assign a label to a region. We can use Gini impurity to measure it.

# Gini Impurity

Let us assume $y$ takes values $0,\ldots, K-1$ i.e. $K$ classes. For a node $m$ and class $k$, we can write probabilities $p_{mk}$ as:
$$p_{mk} = \frac{1}{n_m} \sum_{y \in Q_m} I(y = k)$$
where
- $n_m$ is the number of samples on node $m$
- $I(\cdot)$ is the indicator function

To denote impurity considering all classes at the node $m$, we can write
$$H(Q_m) = \sum_{k=1}^{K} \hat{p}_{mk}(1 - \hat{p}_{mk})$$

> Let us write some mathematical notations to denote how we split into regions. When do we say we have sufficient "confidence"? One way to mesure is to use Gini impurity.

# Quality of split

We have not explained how to grow the tree using impurity function. We can write what will be impurity after choosing $\theta=(j, t_m)$ which shows the quality of split

$$G(Q_m, \theta) = \frac{n_m^{left}}{n_m} H(Q_m^{left}(\theta)) + \frac{n_m^{right}}{n_m} H(Q_m^{right}(\theta))$$

Where
- $n_m^{left}$ are number of samples on left child node
- $n_m^{right}$ are number of samples on right child node
- $n_m$ are number of samples on the node $m$
- $H(\cdot)$ is the impurity function (e.g., Gini impurity).


$G(Q_m, \theta)$ can also be thought of as weigted average impurity after the split

To get the best $\theta^*$, we can choose as below

$$\theta^* = \underset{\theta}{\operatorname{argmin}}\, G(Q_m, \theta)$$

<!-- <ques>Is it impurity reduction or impurity?</ques> <imp>It does not matter since impurity of node $m$ does not change with $\theta$</imp> -->

Note that
1. We optimize over $\theta=(j, t_m)$ i.e., over both features $j$ and thresholds $t_m$
2. Above optimization considers impurity and quality of split based on node $m$ and its child nodes rather than entire tree(greedy optimziation). 

Recurse for subsets $Q_m^{left}$ and $Q_m^{right}$ until the maximum allowable depth is reached, $n_m<min_{samples}$ or $n_m=1$.

# Training Algorithm

1. At each node $m$:
    - For every feature $j$ and possible threshold $t$:
        - Compute the impurity of the split $G(Q_m, \theta)$, where $\theta = (j, t)$
    - Select the split $\theta^* = (j^*, t^*)$ that minimizes $G(Q_m, \theta)$
    - Split the data into $Q_m^{left}$ and $Q_m^{right}$ using $\theta^*$
2. Recursively repeat step 1 for the left and right child nodes
3. Stop splitting when a stopping criterion is met (e.g., maximum depth, minimum samples per node, or node is pure)

# Inferencing algorithm

Given a trained decision tree and a new input sample $x$:

1. **Start at the root node.**
2. **At each internal node:**
    - Evaluate the split condition (e.g., "Is feature $j$ less than threshold $t$?").
    - If the condition is true, move to the left child node; otherwise, move to the right child node.
3. **Repeat step 2** until a leaf node $l$ is reached.
4. **At the leaf node $l$:**  
   - For classification, the predicted class is:
     $$\hat{y} = \underset{k \in \{0, \ldots, K-1\}}{\operatorname{argmax}} \; p_{lk}$$
     where $p_{lk}$ is the proportion of class $k$ samples in leaf $l$.

<!-- <note>**Summary:**</note>
Traverse the tree according to the split conditions until a leaf is reached, then output the majority class (classification) or mean value (regression) of that leaf.
 -->
# WebApp

## How WebApp works?
1. A user inputs random seed and submits. This geneates data $(X, y)$.
2. A decision tree classifier(single tree) is trained on the generated data
3. Trained decision tree and node impurity(or weighted average impurity) for each depth is visualized
4. When a user hovers on impurity vs depth plot, a truncated decision tree(tree truncated upto hovered depth) is created using trained decision tree
5. Using truncated tree, data is reclassified to provide insights about how decision regions, split conditions are formed

## Note
1. When user hovers on node $m$ with depth $d$, decision boundaries are visualized for depth=$d$ rather than node $m$(i.e., all nodes with depth=$d$ are considered leaf nodes)
2. Sometimes it can so happen that decision boundary regions dont change with depth, this is possible as tree may predict the same label after the split
3. Node impurity vs Tree depth figure has reversed x and y axis

## Decision Tree Structure figure
1. When you hover on any node $m$, it shows 2 sets of information is visible. 
    1. **split condition**: shows condition used to split the data at node $m$ further e.g.: `X0 <= 0.7` 
        1. This could just also display `Leaf` if one hovers on leaf node as it will not have any split condition
    2. **samples and prediction**: For the data at node $m$, how many samples belong to each class in their ground truths and what will be the prediciton at that node e.g.: `samples == [3, 8], predict=1`. For all `11 samples`, node $m$ predicts them as class `1`
2. Entire tree structure obtained is visualized for the given data
3. Note that effect of split condition is not associated with prediction in that node
4. Every node shows a condition, when a new example arrives if condition is true then new example is sent to left child node

## Decision regions of subtree figure
1. This shows decision regions considering a given truncated tree
2. Decision regions are shown with lighter color of corresponding samples

## Node impurity vs Tree depth figure

1. Hovering on the impurity vs depth plot at depth $d$ should update the Decision regions of subtree figure to show the regions defined by the tree truncated at depth $d$.
2. if one hovers on a depth $d$ then following are to be considered
    1. The decision boundary should reflect all splits from the root to depth $d$
    2. Classification corresponding to that depth are visualized without considering **split condition of nodes at that depth** (these conditions are used to grow the tree further). 
    3. In other words, if one hovers on $\text{depth}=d$ then regions belonging to $Q_m^{left}$ and $Q_m^{right}$ (nodes having depth $d$) are not visualized but all regions belonging to node $m$ having depth lesser than $d$
3. Node impurity is computed similar to quality of split defined [here](#quality-of-split) execpt that all nodes, predictions and their corresponding data are considered
4. Depth is number of levels in the tree. For root node, Depth = $0$
    1. Number of levels is defined as depth + 1
4. Leaf nodes dont have split condition
5. Plot ticks are aligned with node depth in Decision Tree structure figure

## Instance 1

A simple demonstration of WebApp is:![Instance 1](Simple1.gif)
- We can see how decision tree classifies easier samples on left first correctly and goes deeper to classify harder examples on right

## Instance 2
- Impurity need not reach $0$ at leaf nodes based on various hyperparameters(in this case, `max_depth=7`) and data
![Instance 2](./NonZeroImpurity1,%20200%20samples.gif)

## Observations:
1. We can see that node impurity decreseas with depth
2. Root node classifies entire data into a single class(majrity class) which is expected
3. One can see that algorithm is trying to reduce impurity as much as possible based on the decisions made at each node
4. Impurity need not reach $0$ at leaf nodes

# <note>Complexity</note>

# References

- [scikit-learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation)
- [CART: Classification and Regression Trees (Breiman et al., 1984)](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-richard-olshen-charles-stone)

# Questions
