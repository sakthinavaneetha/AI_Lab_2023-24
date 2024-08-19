# AI_Lab_2023-24
LAB EXPERIMENTS 
REQUIREMENTS 
1. PYHTON
2. SWI-PROLOG
3. PDDL 0R PDDL EDITOR ONLINE
4. JUPITER OR COLAB NOTEBOOK

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


![image](https://github.com/user-attachments/assets/2288b1da-b4ed-43cc-b74a-77cf4515f6f5)
