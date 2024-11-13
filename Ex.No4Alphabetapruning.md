# Ex.No: 4   Implementation of Alpha Beta Pruning 
### DATE:22.08.2024                                                                 
### REGISTER NUMBER : 212222040238
### AIM: 
Write a Alpha beta pruning algorithm to find the optimal value of MAX Player from the given graph.
### Steps:
1. Start the program
2. Initially  assign MAX and MIN value as 1000 and -1000.
3.  Define the minimax function  using alpha beta pruning
4.  If maximum depth is reached then return the score value of leaf node. [depth taken as 3]
5.  In Max player turn, assign the alpha value by finding the maximum value by calling the minmax function recursively.
6.  In Min player turn, assign beta value by finding the minimum value by calling the minmax function recursively.
7.  Specify the score value of leaf nodes and Call the minimax function.
8.  Print the best value of Max player.
9.  Stop the program. 

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

![image](https://github.com/user-attachments/assets/a33e3c17-b6d3-4e54-868d-5fe412d82efe)


### Result:
Thus the best score of max player was found using Alpha Beta Pruning.
