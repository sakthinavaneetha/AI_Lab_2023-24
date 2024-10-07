# Ex.No: 3  Implementation of Minimax Search
### DATE:                                                                            
### REGISTER NUMBER : 
### AIM: 
Write a mini-max search algorithm to find the optimal value of MAX Player from the given graph.
### Algorithm:
1. Start the program
2. import the math package
3. Specify the score value of leaf nodes and find the depth of binary tree from leaf nodes.
4. Define the minimax function
5. If maximum depth is reached then get the score value of leaf node.
6. Max player find the maximum value by calling the minmax function recursively.
7. Min player find the minimum value by calling the minmax function recursively.
8. Call the minimax function  and print the optimum value of Max player.
9. Stop the program. 

### Program:

```
graph={
    '5':['3','7'],
    '3':['2','4'],
    '7':['8'],
    '2':[],
    '4':['8'],
    '8':[]
}
visited = [] 
queue = []
def bfs(visited, graph, node): 
        visited.append(node)
        queue.append(node)
        while queue:
            m = queue.pop(0)
            print(m)
            for neighbour in graph[m]:
                    if neighbour not in visited:
                         visited.append(neighbour)
                         queue.append(neighbour)
                                           
#Driver code
print("Following is the Breadth First Search")
bfs(visited, graph, '5')
```









### Output:

![image](https://github.com/user-attachments/assets/a482cc17-bacc-4ed5-a08b-1256af562ab3)


### Result:
Thus the optimum value of max player was found using minimax search.
