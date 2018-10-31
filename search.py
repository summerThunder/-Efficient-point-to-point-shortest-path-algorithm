import math
import numpy as np
import indexPriorityQueue as ipq
import matplotlib.pyplot as plt
#!!!!!!!!!!
#A星终止条件，终点加入explored,因为最后一步最小的waited必定是真是dist最小的点
#点结构为{id1:(x1,y1)}
#边结构为[(id1,id2)....]
vertexs={}
edges=[]
#启发函数直接用id计算
def h(id1,id2,mode='Euclidean'):
    if mode=='Euclidean':
        return math.sqrt(math.pow(vertexs[id1][0]-vertexs[id2][0],2)
                         +math.pow(vertexs[id1][1]-vertexs[id2][1],2))
    elif mode=='zero':
        return 0
def  make_graph(edges):
    #图用邻接表，结构用列表套字典[{id2:dist02,id3:dist03},{id3:dist13}]
    graph=[]
    for i in range(len(vertexs)):
        graph.append({})

    for edge in edges:

        graph[edge[0]][edge[1]]=h(edge[0],edge[1])
        graph[edge[1]][edge[0]]=h(edge[0],edge[1])

    return graph


def relax(point,waited,index,dist,disth,way,visited,mode,tag):

    for related in graph[point]:
        if dist[point] + graph[point][related] < dist[related]:
            if visited[related] ==0:
                visited[related]=tag
            dist[related] = dist[point] + graph[point][related]
            ipq.replace(waited, index, (disth[related], way[related], related),
                        (dist[related] + h(related, end, mode), point, related))
            disth[related] = dist[related] + h(related, end, mode)
            way[related] = point

def A_star(graph,start,end,mode):
    #调整后的边长度为l(v,w
    visited=np.zeros(len(vertexs),dtype=int)
    #不用索引优先，用字典效率也不差,问题就是可能相同长度重复了,key用三元值。。。那直接用set吧
    #见鬼python的set,字典放入元组时不默认按第一元排序,一般都是按第一元(内在不是红黑树）
    waited=[]
    index={}
    visited[start]=1
    #v是当前visit的点
    v=start
    dist=np.ones(len(vertexs))*9999999
    disth=np.ones(len(vertexs))*9999999
    disth[start]=h(start,end,mode)
    dist[start]=0
    #记录路径的数组,表示前一个的id
    way=np.ones(len(vertexs),dtype=int)*(-1)

    # explored=set()
    # explored.add(start)


    for w in graph[v]:
        if visited[w] == 0:
            visited[w] = 1
            dist[w] = graph[v][w] + dist[v]

            # 一加一减等价于dist[w]+h(w,end)
            disth[w] = dist[w] + h(w, end, mode)
            # waited用set最好
            ipq.push(waited,index,(disth[w], v, w))
            way[w] = v

    while v!=end:

        top=ipq.pop(waited,index)

        v=top[2]
        relax(v,waited,index,dist,disth,way,visited,mode,1)




    return (dist,way,visited)

def bidirection_A_star(graph,start,end,mode):
    visited = np.zeros(len(vertexs), dtype=int)
    waitedf = []
    waitedr=[]
    indexf = {}
    indexr={}
    #1是正向waited,2是逆向waited,3是正向explored,4是反向explored
    visited[start]=3
    visited[end]=4
    # v是当前正向的点，u是当前反向的点
    v = start
    u=end
    distf= np.ones(len(vertexs)) * 9999999
    distr = np.ones(len(vertexs)) * 9999999
    distfh = np.ones(len(vertexs)) * 9999999
    distrh = np.ones(len(vertexs)) * 9999999
    distfh[start] = h(start, end, mode)
    distrh[end]=h(start, end, mode)
    distf[start] = 0
    distr[end]=0
    # 记录路径的数组,表示前一个的id
    wayf = np.ones(len(vertexs), dtype=int) * (-1)
    wayr = np.ones(len(vertexs), dtype=int) * (-1)
    for w in graph[v]:
        if visited[w] == 0 or visited[w] == 2:
            visited[w] = 1
            distf[w] = graph[v][w] + distf[v]
            distfh[w] = distf[w] + h(w, end, mode)
            ipq.push(waitedf,indexf,(distfh[w], v, w))
            wayf[w] = v
    for l in graph[u]:
        if visited[l] == 0 or visited[l] == 1:
            visited[l] = 2
            distr[l] = graph[u][l] + distr[u]
            distrh[l] = distr[l] + h(l, start, mode)
            ipq.push(waitedr,indexr,(distfh[l], u, l))
            wayr[l] = u

    while 1:
        topf = ipq.pop(waitedf, indexf)
        v = topf[2]
        relax(v,waitedf,indexf,distf,distfh,wayf,visited,mode,1)
        print(waitedr)
        topr = ipq.pop(waitedr, indexr)
        print(waitedr)
        u = topr[2]
        relax(u,waitedr,indexr,distr,distrh,wayr,visited,mode,2)
        if visited[v]!=4:
            visited[v]=3
        else:
            return (v,distf,distr,wayf,wayr,visited)
        if visited[u]!=3:
            visited[u]=4
        else:
            return (u,distf,distr,wayf,wayr,visited)



# def plot_way(vertexs,way):
#     global plt
#     for i in range(len(way)):
#         if way[i]!=-1:
#             plt.plot([vertexs[i][0],vertexs[way[i]][0]],[vertexs[i][1],
#                                                         vertexs[way[i]][1]],c='g')
#     plt.show()

def test1(mode):
    dist, way, visited = A_star(graph, start, end, mode)
    print('A_star结束')
    for j in range(len(way)):
        if way[j] != -1:
            p1 = j
            p2 = way[j]

            plt.plot([vertexs[p1][0], vertexs[p2][0]], [vertexs[p1][1], vertexs[p2][1]], c='b')
    k = end

    while k != -1:
        k_last = way[k]
        print(k)
        color = 'r'
        if k == end or k == start:
            color = 'b'
        plt.scatter(vertexs[k][0], vertexs[k][1], s=80, c=color)
        if k_last != -1:
            plt.plot([vertexs[k_last][0], vertexs[k][0]], [vertexs[k_last][1], vertexs[k][1]], c='r')
        k = k_last
    print('画出路径图')
    print('最短路径长度是', dist[end])
    print('扫描过的点有', np.sum(visited))

    plt.show()

def test2(mode):
    u, distf, distr, wayf, wayr, visited=bidirection_A_star(graph,start,end,mode)
    for j in range(len(wayf)):
        if wayf[j] != -1:
            p1 = j
            p2 = wayf[j]
            plt.plot([vertexs[p1][0], vertexs[p2][0]], [vertexs[p1][1], vertexs[p2][1]], c='b')

    for j in range(len(wayr)):
        if wayr[j] != -1:
            p1 = j
            p2 = wayr[j]

            plt.plot([vertexs[p1][0], vertexs[p2][0]], [vertexs[p1][1], vertexs[p2][1]], c='g')

    k = u
    print(u)

    while k != -1:
        k_last = wayf[k]
        print(k)
        color = 'r'
        if k == end or k == start:
            color = 'b'
        plt.scatter(vertexs[k][0], vertexs[k][1], s=80, c=color)
        if k_last != -1:
            plt.plot([vertexs[k_last][0], vertexs[k][0]], [vertexs[k_last][1], vertexs[k][1]], c='r')
        k = k_last

    k = u

    while k != -1:
        k_last = wayr[k]
        print(k,visited[k])
        color = 'r'
        if k == end or k == start:
            color = 'b'
        plt.scatter(vertexs[k][0], vertexs[k][1], s=80, c=color)
        if k_last != -1:
            plt.plot([vertexs[k_last][0], vertexs[k][0]], [vertexs[k_last][1], vertexs[k][1]], c='r')
        k = k_last
    plt.show()

    
if __name__=='__main__':
    np.random.seed(55)
    rand1=np.random.rand(300,2)*100
    rand1=np.unique(rand1,axis=0)
    plt.scatter(rand1[:,0],rand1[:,1])

    for i in range(len(rand1)):
        vertexs[i]=(rand1[i,0],rand1[i,1])
    np.random.seed(55)
    rand2=np.random.randint(300,size=(2500,2))
    for i in range(2500):
        p1=rand2[i,0]
        p2=rand2[i,1]
        if h(p1,p2,'Euclidean')<30:
            plt.plot([rand1[p1,0],rand1[p2,0]],[rand1[p1,1],rand1[p2,1]],c='gray')
            edges.append(rand2[i,:])


    graph=make_graph(edges)

    start=10
    end=139
    test2('Euclidean')
