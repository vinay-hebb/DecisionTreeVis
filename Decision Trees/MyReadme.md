<style>
  ques { background-color: red; color: white; padding: 1px; border-radius: 2px; }
  note { background-color: yellow; color: black; padding: 1px; border-radius: 2px; }
  imp { background-color: lightgreen; color: black; padding: 1px; border-radius: 2px; }
  underline1 { text-decoration: underline; }
</style>

# Basic notations:
- At any node, When a split condition is met then, we traverse to left subtree
- Each node has
    1. #Samples present in the node
    2. Split condition

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
- <ques>Decision boundary can be visualized for each level?</ques>
- <ques>In impurity vs depth, when root node and children hovered, it shows same decision boundary, why?</ques>
- <ques>When we hover on DecBound(level=1), it should show 2 regions - it is not happening</ques>


https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation