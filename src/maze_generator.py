#Generates a random maze with BFS search implemented for two reasons: 1) Checks if the maze is solvable
#                                                                     2) Give the shortest path to the goal


import numpy as np
from collections import deque

def bfs_shortest_path_length(maze: np.ndarray, start, goal):
    
    rows, cols = maze.shape
    
    #Tuples are immutable and this way it could be used as key/value pair
    start = tuple(start)
    goal = tuple(goal)

    #If the wall is at the position of the start or the goal the maze is not solvable
    if maze[start] == 1 or maze[goal]==1:
        return None
    

    visited = np.zeros_like(maze, dtype = bool)

    #queue is used as a deque of the visited position with the information of the distance from the start. 
    #This guarantees the shortes path to the goal. For each cell we visit the neighbouring ones
    queue = deque()
    queue.append((start, 0))
    visited[start] = True

    moves = [(-1,0), (1, 0), (0, -1), (0, 1)]


    """
    The point of the following loop is as follows: We start at the postion (0,0), 0 tiles away from the start.
    Then, we visit the neighbouring ones, while popping the starting position from deque. That way we move "layer-by-layer" or "cell-by-cell" through the maze.
    If the goal can not be reached, in the last iteration, the deque will be empty (all the postions will be popped from the deque) and the loop finishes and returns None
    """

    while queue:

        (r, c), dist = queue.popleft()

        if (r, c) == goal:
            return dist


        for dr, dc in moves:
            nr, nc = r+dr, c+dc

            if 0<= nr < rows and 0 <= nc <cols and maze[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append(((nr, nc), dist+1))


    return None



def _generate_maze(size: int=10, wall_density: float=0.3, rng: np.random.Generator = None, min_path_length: int=1, max_attempts: int=1000):
    
    if rng is None:
        rng = np.random.default_rng()

    for attempt in range(max_attempts):
        
        maze = (rng.random((size, size)) < wall_density).astype(np.int32)
        empty_cells = np.argwhere(maze == 0)
        if len(empty_cells) < 2:
            continue  # too few empty cells, retry
        # Pick two distinct empty cells
        idx = rng.choice(len(empty_cells), size=2, replace=False)
        start = tuple(empty_cells[idx[0]])
        goal = tuple(empty_cells[idx[1]])

        # 3. Check solvability with BFS
        path_len = bfs_shortest_path_length(maze, start, goal)

        if path_len is not None and path_len >= min_path_length:
            return {
                "maze": maze,
                "start": np.array(start, dtype=np.int32),
                "goal": np.array(goal, dtype=np.int32),
                "optimal_len": path_len,
            }

    raise RuntimeError(
        f"Failed to generate a solvable maze after {max_attempts} attempts. "
        f"Try lowering wall_density (currently {wall_density})."
    )


def generate_maze_set(count: int, size:int = 10, wall_density: float=0.3, seed: int=42, min_path_length:int = 4):
    rng = np.random.default_rng(seed)
    mazes=[]
    for i in range(count):
        maze_data = _generate_maze(size=size, wall_density=wall_density, rng = rng, min_path_length = min_path_length)
        mazes.append(maze_data)

    return mazes



    

    



