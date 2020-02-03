#following: https://julie-jiang.github.io/image-segmentation/

from Pillow import Image
import numpy as np
import math
from collections import deque

class Vertex:
    def __init__(self, n, data=None, above=None, below=None, left=None, right=None):
        self.above = above
        self.below = below
        self.left = left
        self.right = right
        self.name = n
        self.data = data
        self.visited = False
    
    def get_data(self):
        return self.data
    
    def get_name(self):
        return self.name

    def __str__(self):
        return str((self.name,self.data))

class Graph:
    vertices = {}
    edges = []
    vertex_indices = {}
    index_to_vertex = {}

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex) and vertex.name not in self.vertices:
            self.vertices[vertex.name] = vertex
            for row in self.edges:
                row.append(0)
            self.edges.append([0] * (len(self.edges)+1))
            self.vertex_indices[vertex.name] = len(self.vertex_indices) #since we go from a 2d representation of nxm pixels to a graph that's pxp, we need to map everything in nxm -> p; that mapping is assigned here
            self.index_to_vertex[len(self.vertex_indices)-1] = vertex.name
            return True
        return False
    
    def add_edge(self, u, v, weight=0):
        ''' Adds an undirected edge (more precisely, a pair of directed edges)
            Input: u,v - keys in vertices dict (name)
            Output: whether or not the edge could be added '''
        if u in self.vertices and v in self.vertices:
            self.edges[self.vertex_indices[u]][self.vertex_indices[v]] = weight
            self.edges[self.vertex_indices[v]][self.vertex_indices[u]] = weight
            return True
        return False
    
    def add_directed_edge(self, u, v, weight=0):
        ''' Adds a directed edge
            Input: u,v - keys in vertices dict (name)
            Output: whether or not the edge could be added '''
        if u in self.vertices and v in self.vertices:
            self.edges[self.vertex_indices[u]][self.vertex_indices[v]] = weight
            return True
        return False
    
    def print_graph(self):
        for v, i in sorted(self.vertex_indices.items()):
            print v + ' ',
            for j in range(len(self.edges)):
                print self.edges[i][j], 
            print(' ')    
    
def penalty(u, v, sigma):
    ''' Computes boundary penalty given two Vertex objects
        Input: u - a Vertex object
               v - a Vertex object
        Output: boundary penalty - int
    '''
    return int(100*math.exp((-(u.get_data()-v.get_data())**2)/(2*sigma**2)))

def intensity(rgb):
    '''Gets the intensity of an rgb array'''
    return 0.2126 * rgb[0] ** 2.2 + 0.7152 * rgb[1] ** 2.2 + 0.0722 * rgb[2] ** 2.2

def image_to_array(filename):
    '''Converts image to np array'''
    im = Image.open(filename, 'r')
    return np.asarray(im)

def add_vertices(graph, image_array):
    im = image_array
    # if isinstance(im, np.ndarray):
    width, height = im.shape[1], im.shape[0]
    for row in range(0,height):
        for col in range(0,width):
            v = Vertex((row,col), intensity(im[row][col]))
            graph.add_vertex(v)
    #     return True
    # return False
 
def add_edges(graph, image_array):
    im = image_array
    # if isinstance(im, np.ndarray):
    width, height = im.shape[1]-1, im.shape[0]-1
    for vertex_pos in graph.vertices.keys():
        if vertex_pos[0] < height:
            graph.add_edge(vertex_pos, (vertex_pos[0]+1, vertex_pos[1]), penalty(graph.vertices[vertex_pos],graph.vertices[(vertex_pos[0]+1, vertex_pos[1])], 100000))
        if vertex_pos[1] < width:
            graph.add_edge(vertex_pos, (vertex_pos[0], vertex_pos[1]+1), penalty(graph.vertices[vertex_pos],graph.vertices[(vertex_pos[0], vertex_pos[1]+1)], 100000))
        if vertex_pos[0] > 0:
            graph.add_edge(vertex_pos, (vertex_pos[0]-1, vertex_pos[1]), penalty(graph.vertices[vertex_pos],graph.vertices[(vertex_pos[0]-1, vertex_pos[1])], 100000))
        if vertex_pos[1] > 0:
            graph.add_edge(vertex_pos, (vertex_pos[0], vertex_pos[1]-1), penalty(graph.vertices[vertex_pos],graph.vertices[(vertex_pos[0], vertex_pos[1]-1)], 100000))
    #     return True
    # return False

def add_source_sink(graph, source, sink):
    s = Vertex("Source")
    t = Vertex("Sink")
    graph.add_vertex(s)
    graph.add_vertex(t)

    for edge in source:
        #edge[0] describes the name of the neighboring vertex
        #edge[1] describes the weight, set by default to the maximum, of 100
        graph.add_directed_edge("Source", edge[0], edge[1])
    
    for edge in sink:
        #edge[0] describes the name of the neighboring vertex
        #edge[1] describes the weight, set by default to the maximum, of 100
        graph.add_directed_edge(edge[0], "Sink", edge[1])


def create_graph(graph, filename, source, sink):
    graph = Graph()
    image_array = image_to_array(filename)
    add_vertices(graph, image_array)
    add_edges(graph, image_array)
    add_source_sink(graph, source, sink)

source = [((64,28),10000),((67,78),10000)]#[((258, 183),10000), ((297,210),10000), ((207,294),10000), ((448,252),10000)]
sink = [((15,35),10000),((13,70),10000)]#[((1,4),10000)]

graph1 = Graph()
create_graph(graph1, "C:/Users/prana/Documents/image_segmentation/handwriting/test1.jpeg", source, sink)
#for key in graph1.vertex_indices.keys():
#    print((key,graph1.vertex_indices[key])) #since we go from a 2d representation of nxm pixels to a graph that's pxp, we need to map everything in nxm -> p; that mapping is recorded here
#for row in graph1.edges:
#    print(row)

#print

#print graph1.edges[-1]
#print graph1.edges[-2]
#print [x[-1] for x in graph1.edges]

residualGraph = [[0 for x in range(len(graph1.edges))] for y in range(len(graph1.edges))]

def dfs(graph, resGraph):
    #remember that the row signifies the "pointer" and the column signifies the "pointee"
    edges = graph.edges
    verts = graph.vertices
    source = (edges[-2], len(edges)-2)
    stack = []
    pathTracker = {}
    stack.append(source)
    verts[graph.index_to_vertex[len(edges)-2]].visited = True
    allVisited = [verts[graph.index_to_vertex[len(edges)-2]]]

    while len(stack) > 0:
        top = stack.pop()
        for i in range(len(top[0])):
            #going through a list of neighbors, which looks like [0, 100, 0, 14, 0, etc...]
            if top[0][i] is 0:
                continue
            if verts[graph.index_to_vertex[i]].name is 'Sink':
                pathTracker[i] = top[1]
                break
            if not verts[graph.index_to_vertex[i]].visited:
                if edges[top[1]][i] - resGraph[top[1]][i] > 0: #see if there's flow left
                    stack.append((edges[i], i))#if so, push it
                    pathTracker[i] = top[1] #inspiration from here: https://stackoverflow.com/questions/12864004/tracing-and-returning-a-path-in-depth-first-search
                    verts[graph.index_to_vertex[i]].visited = True #mark visited after pushing
                    allVisited.append(verts[graph.index_to_vertex[i]])
    
    for x in allVisited: #clear visited, could've used a visited matrix as that would clear as soon as the method ended execution...
        x.visited = False
    
    path = []
    current = len(edges)-1 #inspiration from here: https://stackoverflow.com/questions/12864004/tracing-and-returning-a-path-in-depth-first-search
    
    if current not in pathTracker:
        return path
    
    while (current != len(edges)-2):
        path.append(graph.index_to_vertex[current])
        current = pathTracker[current]
    path.append(graph.index_to_vertex[len(edges)-2])

    return path


path = dfs(graph1, residualGraph)

def edmondsKarp(graph, resGraph):
    path = dfs(graph, resGraph) #this gets you the path, but its reversed. We handle that in the min statement, by switching the order of the vertex indices when finding edge weights (i.e. graph.edges[graph.vertex_indices[path[x+1]]][graph.vertex_indices[path[x]]])
    
    while('Sink' in path):
        edges = [(graph.vertex_indices[path[x+1]],graph.vertex_indices[path[x]]) for x in range(len(path)-1)]
        edges.reverse()
        minCap = min([graph.edges[edge[0]][edge[1]] - resGraph[edge[0]][edge[1]] for edge in edges])#min([graph.edges[graph.vertex_indices[path[x+1]]][graph.vertex_indices[path[x]]] - resGraph[graph.vertex_indices[path[x+1]]][graph.vertex_indices[path[x]]] for x in range(len(path)-1)])
        
        for edge in edges:
            #(u,v) += minCap
            resGraph[edge[0]][edge[1]] += minCap
            #(v,u) -= minCap
            resGraph[edge[1]][edge[0]] -= minCap
        
        path = dfs(graph, resGraph)

edmondsKarp(graph1, residualGraph)

def findCut(graph, resGraph):
    #bft from source to all edges that arent at capacity, making huge list of vertices/pixels that are in the source segment
    pixels = [graph.vertices[graph.index_to_vertex[len(graph.edges)-2]]]
    #wheele
    q = []
    source = (graph.edges[-2],len(graph.edges)-2)
    q.append(source)

    edges = [(graph.vertex_indices[path[x+1]],graph.vertex_indices[path[x]]) for x in range(len(path)-1)]
    edges.reverse()    

    while len(q) > 0:
        top = q.pop(0)
        #print top
        for i in range(len(top[0])):
            if top[0][i] is not 0:
                if top[0][i] - resGraph[top[1]][i] > 0:#graph.edges[top[1]][i] - resGraph[top[1]][i] > 0:
                    if not graph.vertices[graph.index_to_vertex[i]].visited:
                        pixels.append(graph.vertices[graph.index_to_vertex[i]])
                        q.append((graph.edges[i], i))#if so, push it
                        graph.vertices[graph.index_to_vertex[i]].visited = True #mark visited after pushing
    
    return pixels

print "\n\n\n"


#finally, the bft that gets all vertices associated with pixels in the source segment/foreground
source_pixels = findCut(graph1, residualGraph)

print [str(x) for x in source_pixels]

#finally, create and save an image that has a blue sink segment (a blue background) and a red source segment (a red foreground)
segmented = Image.new('RGB', Image.open("C:/Users/prana/Documents/image_segmentation/handwriting/imeg.jpeg",'r').size, color = (0, 0, 255))
for i in source_pixels:
    if not i.name == "Source":
        segmented.putpixel(i.name, (255, 0, 0))
segmented.save("imeg_segmented.png")

