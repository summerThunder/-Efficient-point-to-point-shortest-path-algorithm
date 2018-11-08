import math
import numpy as np
import indexPriorityQueue as ipq
import matplotlib.pyplot as plt
import threading
import time
#!!!!!!!!!!
#A星终止条件，终点加入explored,因为最后一步最小的waited必定是真是dist最小的点
#点结构为{id1:(x1,y1)}
#边结构为[(id1,id2)....]
vertexs={}
edges=[]
#启发函数直接用id计算
def h(id1,id2,mode,id3=-1):
    if mode=='Euclidean':
        return math.sqrt(math.pow(vertexs[id1][0]-vertexs[id2][0],2)
                         +math.pow(vertexs[id1][1]-vertexs[id2][1],2))
    elif mode=='zero':
        return 0
    elif mode=='bidirection':
        return (math.sqrt(math.pow(vertexs[id1][0]-vertexs[id2][0],2)
                         +math.pow(vertexs[id1][1]-vertexs[id2][1],2)) -math.sqrt(math.pow(vertexs[id1][0]-vertexs[id3][0],2)
                         +math.pow(vertexs[id1][1]-vertexs[id3][1],2)))*0.5

def  make_graph(edges):
    #图用邻接表，结构用列表套字典[{id2:dist02,id3:dist03},{id3:dist13}]
    graph=[]
    for i in range(len(vertexs)):
        graph.append({})

    for edge in edges:

        graph[edge[0]][edge[1]]=h(edge[0],edge[1],'Euclidean')
        graph[edge[1]][edge[0]]=h(edge[0],edge[1],'Euclidean')

    return graph


# def relax(point,waited,index,dist,disth,way,mode,start,end,shortest):
#     for related in graph[point]:
#         if dist[point] + graph[point][related] < dist[related]:
#
#
#             dist[related] = dist[point] + graph[point][related]
#             ipq.replace(waited, index, (disth[related], way[related], related),
#                         (dist[related] + h(related, end, mode,start), point, related))
#             disth[related] = dist[related] + h(related, end, mode,start)
#             way[related] = point
#
#             print(point,related)
#             if visited[related] == advtag and shortest>

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



    for w in graph[v]:
        if visited[w] == 0:
            visited[w] = 1
            dist[w] = graph[v][w] + dist[v]

            # 一加一减等价于dist[w]+h(w,end)
            disth[w] = dist[w] + h(w, end, mode,start)
            ipq.push(waited,index,(disth[w], v, w))
            way[w] = v

    while v!=end:

        top=ipq.pop(waited,index)
        v=top[2]
        for w in graph[v]:
            if dist[v]+graph[v][w]<dist[w]:
                if visited[w]==0:
                    visited[w]=1
                dist[w]= dist[v]+graph[v][w]
                ipq.replace(waited,index,(disth[w],way[w],w),(dist[w]+h(w,end,mode,start),v,w))
                disth[w]=dist[w]+h(w,end,mode,start)
                way[w]=v


    return (dist,way,visited)

def bidirection_A_star(graph,start,end,mode):
    visited = np.zeros(len(vertexs), dtype=int)
    waitedf = []
    waitedr=[]
    indexf = {}
    indexr={}
    visited[start]+=1
    visited[end]+=2
    # v是当前正向的点，u是当前反向的点
    v = start
    u=end
    distf= np.ones(len(vertexs)) * 9999999
    distr = np.ones(len(vertexs)) * 9999999
    distfh = np.ones(len(vertexs)) * 9999999
    distrh = np.ones(len(vertexs)) * 9999999
    distfh[start] = h(start, end, mode,start)
    distrh[end]=h(end,start , mode,end)
    distf[start] = 0
    distr[end]=0
    # 记录路径的数组,表示前一个的id
    wayf = np.ones(len(vertexs), dtype=int) * (-1)
    wayr = np.ones(len(vertexs), dtype=int) * (-1)
    shortest = 99999999
    intersect = -1
    for w in graph[v]:
        visited[w]+=1
        distf[w] = graph[v][w] + distf[v]
        distfh[w] = distf[w] + h(w, end, mode,start)
        ipq.push(waitedf,indexf,(distfh[w], v, w))
        wayf[w] = v



    for l in graph[u]:
        visited[l] += 2
        distr[l] = graph[u][l] + distr[u]
        distrh[l] = distr[l] + h(l, start,mode,end)
        ipq.push(waitedr,indexr,(distrh[l], u, l))
        wayr[l] = u
        if visited[l]==3 and distfh[l]+distrh[l]<shortest:
            shortest=distfh[l]+distrh[l]
            intersect=l
            # print('找到一条路径',intersect,shortest)

    def forward():
        nonlocal waitedf,indexf,distf,distfh,visited,wayf,shortest, intersect,v
        # print('f', waitedf)

        topf = ipq.pop(waitedf, indexf)
        # print(topf)
        v = topf[2]
        for w in graph[v]:
            if distf[v] + graph[v][w] < distf[w]:
                distf[w] = distf[v] + graph[v][w]
                ipq.replace(waitedf, indexf, (distfh[w], wayf[w], w), (distf[w] + h(w, end, mode, start), v, w))
                distfh[w] = distf[w] + h(w, end, mode, start)
                wayf[w] = v

                if visited[w] != 1 and visited[w] != 3:
                    visited[w] += 1

                if visited[w] == 3 and distfh[w] + distrh[w] < shortest:
                    shortest = distfh[w] + distrh[w]
                    intersect = w
                    # print('找到一条更短路径', intersect, shortest)

    def backward():
        nonlocal waitedr,indexr,distr,distrh,visited,wayr,shortest,intersect,u
        # print('r', waitedr)
        topr = ipq.pop(waitedr, indexr)
        # print(topr)
        u = topr[2]
        for l in graph[u]:
            if distr[u] + graph[u][l] < distr[l]:
                distr[l] = distr[u] + graph[u][l]
                ipq.replace(waitedr, indexr, (distrh[l], wayr[l], l), (distr[l] + h(l, start, mode, end), u, l))
                distrh[l] = distr[l] + h(l, start, mode, end)
                wayr[l] = u

                if visited[l] != 2 and visited[l] != 3:
                    visited[l] += 2

                if visited[l] == 3 and distfh[l] + distrh[l] < shortest:
                    shortest = distfh[l] + distrh[l]
                    intersect = l
                    # print('找到一条更短路径', intersect, shortest)


    while distfh[v]+distrh[u]<shortest:
        # t1=threading.Thread(target=forward)
        # t2 = threading.Thread(target=forward)
        # t1.start()
        # t2.start()
        # t1.join()
        # t2.join()

        forward()
        backward()





        # print('top和',distfh[v]+distrh[u],shortest)
        # print('id138',distfh[138],distrh[138],visited[138])
        # print('f',waitedf)
        # print('r',waitedr)


        #画图
        # exploredf.append((topf[1],topf[2]))
        # exploredr.append((topr[1], topr[2]))
        # plot_way(exploredf,exploredr,'b','g')



    return (intersect, shortest,distf, distr, wayf, wayr, visited)


def plot_way(exploredf,exploredr,c1,c2):
    for i in range(2500):
        p1=rand2[i,0]
        p2=rand2[i,1]
        if h(p1,p2,'Euclidean')<30:
            plt.plot([rand1[p1,0],rand1[p2,0]],[rand1[p1,1],rand1[p2,1]],c='gray')
            edges.append(rand2[i,:])
    for i in exploredf:
        p1=vertexs[i[0]]
        p2=vertexs[i[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=c1)
    for i in exploredr:
        p1 = vertexs[i[0]]
        p2 = vertexs[i[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=c2)
    plt.show()




def test1(mode):
    start_time=time.time()
    dist, way, visited = A_star(graph, start, end, mode)
    end_time=time.time()
    print('A_star结束')
    for j in range(len(way)):
        if way[j] != -1:
            p1 = j
            p2 = way[j]

            plt.plot([vertexs[p1][0], vertexs[p2][0]], [vertexs[p1][1], vertexs[p2][1]], c='b')
    k = end

    while k != -1:
        k_last = way[k]

        # print(k)
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
    print('耗时',end_time-start_time)

    plt.show()

def test2(mode):
    start_time = time.time()
    intersect,shortest, distf, distr, wayf, wayr, visited=bidirection_A_star(graph,start,end,mode)
    end_time=time.time()
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

    k = intersect
    # print('交点',intersect,shortest)

    while k != -1:
        k_last = wayf[k]
        # print(k,visited[k])
        color = 'r'
        if k == end or k == start:
            color = 'b'
        plt.scatter(vertexs[k][0], vertexs[k][1], s=80, c=color)
        if k_last != -1:
            plt.plot([vertexs[k_last][0], vertexs[k][0]], [vertexs[k_last][1], vertexs[k][1]], c='r')
        k = k_last

    k = intersect

    while k != -1:
        k_last = wayr[k]
        # print(k,visited[k])
        color = 'r'
        if k == end or k == start:
            color = 'b'
        plt.scatter(vertexs[k][0], vertexs[k][1], s=80, c=color)
        if k_last != -1:
            plt.plot([vertexs[k_last][0], vertexs[k][0]], [vertexs[k_last][1], vertexs[k][1]], c='r')
        k = k_last

    sum=0
    for i in visited:
        if i>0:
            sum+=1
    dist=distf[intersect]+distr[intersect]
    print('画出路径图')
    print('最短路径长度是', dist)
    print('扫描过的点有', sum)
    print('耗时',end_time-start_time)
    plt.show()



np.random.seed(55)
rand1=np.random.rand(500,2)*1000
rand1=np.unique(rand1,axis=0)
plt.scatter(rand1[:,0],rand1[:,1])

for i in range(len(rand1)):
    vertexs[i]=(rand1[i,0],rand1[i,1])
np.random.seed(55)
rand2=np.random.randint(500,size=(5500,2))
for i in range(2500):
    p1=rand2[i,0]
    p2=rand2[i,1]
    hc=h(p1,p2,'Euclidean')
    if hc<300 and hc!=0:
        plt.plot([rand1[p1,0],rand1[p2,0]],[rand1[p1,1],rand1[p2,1]],c='gray')
        edges.append(rand2[i,:])


graph=make_graph(edges)

start=485
end=2


test2('zero')

#test1('zero')

#295
#156 双向dij不对
