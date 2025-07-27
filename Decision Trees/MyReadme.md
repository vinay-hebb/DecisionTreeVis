<style>
  ques { background-color: red; color: white; padding: 1px; border-radius: 2px; }
  note { background-color: yellow; color: black; padding: 1px; border-radius: 2px; }
  imp { background-color: lightgreen; color: black; padding: 1px; border-radius: 2px; }
  underline1 { text-decoration: underline; }
</style>
# Decision Tree Classifier: Mathematical Formulation (scikit-learn)


A **Decision Tree Classifier** is a supervised learning algorithm that recursively partitions the feature space to classify data points. The scikit-learn implementation uses the **CART** (Classification and Regression Trees) algorithm, which builds binary trees using the feature and threshold that yield the largest information gain (or impurity reduction).


# Background
Given
- $x_i \in \mathbb{R}^n,\quad i = 1, \ldots, l$
- $y \in \mathbb{R}^l$<ques>Is it l or scalar?</ques>

> A simple way is to split the regions in the hope that set of examples having same label $y$ are closeby.

Goal is to find an estimator which estimates $y$ reliably for a new example $x$. A way to estimate new example's label is by splitting the given region and labelling the entire region as a particular label if we have sufficient confidence. Any new example is assigned that label if it falls in the given region. A computationally simplest way to split into regions is to use thresholds on particular features. To gain confidence of regions, we can split it into many regions recursively. This can be represented as tree with nodes. Each node has a split condition and set of examples associated with it. 

## Notations
- Consider a node $m$ and let feature be $j$ on which we make a split.
- Let us denote $\theta=(j, t_m)$ and $t_m$ is threshold which separates 2 regions
- $Q_m^{left}(\theta)$ and $Q_m^{right}(\theta)$ are 2 regions split by node $m$. Mathematically, they can be written as below

$$
\begin{aligned}
Q_m^{left}(\theta)  & = \{(x, y) \mid x_j \leq t_m\} \\
Q_m^{right}(\theta) & = Q_m \setminus Q_m^{left}(\theta)
\end{aligned}
$$

We have setup the regions but we have not answered, when do we say we have sufficient "confidence" to assign a label to a region. We can use Gini impurity to measure it.

## 1. Gini Impurity

Let us assume $y$ takes values $0,\ldots, K-1$ i.e. $K$ classes. For a node $m$ and class $k$, we can write probabilities $p_{mk}$ as:
$$p_{mk} = \frac{1}{n_m} \sum_{y \in Q_m} I(y = k)$$
where
- $n_m$ is the number of samples on node $m$
- $I(\cdot)$ is the indicator function

To denote impurity considering all classes at the node $m$, we can write
$$H(Q_m) = \sum_{k=1}^{K} \hat{p}_{mk}(1 - \hat{p}_{mk})$$

> Let us write some mathematical notations to denote how we split into regions. When do we say we have sufficient "confidence"? One way to mesure is to use Gini impurity.

## 2. Quality of split

We have not explained how to grow the tree using impurity function. We can write what will be impurity after choosing $\theta=(j, t_m)$ which shows the quality of split

$$G(Q_m, \theta) = \frac{n_m^{left}}{n_m} H(Q_m^{left}(\theta)) + \frac{n_m^{right}}{n_m} H(Q_m^{right}(\theta))$$

Where
- $n_m^{left}$ are number of samples on left child node
- $n_m^{right}$ are number of samples on right child node
- $n_m$ are number of samples on the node $m$
- $H(\cdot)$ is the impurity function (e.g., Gini impurity).


$G(Q_m, \theta)$ is the weigted average impurity after the split

To get the best $\theta^*$, we can choose as below
$$\theta^* = \operatorname{argmin}_\theta  G(Q_m, \theta)$$

<ques>Is it impurity reduction or impurity?</ques> <imp>It does not matter since impurity of node $m$ does not change with $\theta$</imp>

Note that
1. We optimize over $\theta=(j, t_m)$ i.e., over both features $j$ and thresholds $t_m$
2. Above optimization considers impurity and quality of split based on node $m$ and its child nodes rather than entire tree(greedy optimziation). 

Recurse for subsets $Q_m^{left}$ and $Q_m^{right}$ until the maximum allowable depth is reached, $n_m<min_{samples}$ or $n_m=1$.

## 4. Training Algorithm

1. At each node $m$:
    - For every feature $j$ and possible threshold $t$:
        - Compute the impurity of the split $G(Q_m, \theta)$, where $\theta = (j, t)$
    - Select the split $\theta^* = (j^*, t^*)$ that minimizes $G(Q_m, \theta)$
    - Split the data into $Q_m^{left}$ and $Q_m^{right}$ using $\theta^*$
2. Recursively repeat step 1 for the left and right child nodes
3. Stop splitting when a stopping criterion is met (e.g., maximum depth, minimum samples per node, or node is pure)

## 5. Inferencing algorithm

Given a trained decision tree and a new input sample $x = (x_1, x_2, \ldots, x_p)$:

1. **Start at the root node.**
2. **At each internal node:**
    - Evaluate the split condition (e.g., "Is feature $j$ less than threshold $t$?").
    - If the condition is true, move to the left child node; otherwise, move to the right child node.
3. **Repeat step 2** until a leaf node $l$ is reached.
4. **At the leaf node $l$:**  
   - For classification, the predicted class is:
     $$\hat{y} = \underset{k \in \{0, \ldots, K-1\}}{\operatorname{argmax}} \; p_{lk}$$
     where $p_{lk}$ is the proportion of class $k$ samples in leaf $l$.
   - For regression, the prediction is:
     $$\hat{y} = \frac{1}{n_l} \sum_{i \in Q_l} y_i$$
     where $n_l$ is the number of samples in leaf $l$ and $Q_l$ is the set of samples in $l$.

**Summary:**  
Traverse the tree according to the split conditions until a leaf is reached, then output the majority class (classification) or mean value (regression) of that leaf.

## 6. Explain using demo
1. In Node impurity vs depth subplot, if one hovers on a depth then nodes present in that depth are visualized without considering their split condition which will be used to grow the tree further. In other words, if one hovers on $\text{depth}=d$ then regions belonging to $Q_m^{left}$ and $Q_m^{right}$ are not visualized but all regions belonging to node $m$ having depth $d$

## 7. References

- [scikit-learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation)
- [CART: Classification and Regression Trees (Breiman et al., 1984)](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-richard-olshen-charles-stone)

---


# Others
- When hovered, any node shows
    1. #Samples present in the node. (This is not #samples after the split condition is applied)
    2. Split condition(defines how much samples present in current node should go to each children)
- So, root node shows all samples present in the dataset
- Leaf nodes dont have split condition
- Level := [0, height of tree] - closed interval
- depth + 1 = #levels
- Impurity can be computed for each level
- DecBound(level=0) is entire set

# Questions
- <ques>Decision boundary can be visualized for each depth?</ques>
- <ques>In impurity vs depth, when root node and children hovered, it shows same decision boundary, why?</ques>
    - <ques>When we hover on DecBound(level=1), it should show 2 regions - it is not happening</ques>


https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation