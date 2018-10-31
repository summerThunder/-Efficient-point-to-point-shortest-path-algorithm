
#用于记录index
def change_dict(dict,k1,k2):
    temp=dict[k1]
    dict[k1]=dict[k2]
    dict[k2] = temp
    return dict

#sink不能截取中间一段开始，树形不同
def sink(array,index,stid):
    #与左右子中较小的交换，否则会出现逆序
    l=len(array)
    i=stid
    while(i<int(l/2)):
        min=-1
        #不一定有右节点存在
        if 2*i+2>=l:
            min=2*i+1
        elif array[2*i+1]>=array[2*i+2]:
            min=2*i+2
        else:
            min=2*i+1
        if array[i]>array[min]:
            temp=array[i]
            array[i]=array[min]
            array[min]=temp

            change_dict(index,array[i],array[min])
            i =min
        else:
            break
    return i

def sift(array,index,stid):
    i=stid
    while i>0 and array[i]<array[int((i-1)/2)]:
        temp = array[i]
        array[i] = array[int((i-1)/2)]
        array[int((i-1)/2)] = temp
        change_dict(index,array[i], array[int((i-1)/2)])
        i=int((i-1)/2)

    return i

def push(heap,index,obj):
    heap.append(obj)
    index[obj]=len(heap)-1
    loc=sift(heap,index,len(heap)-1)
    return loc

def pop(heap,index):
    top=heap[0]
    heap[0]=heap[-1]
    change_dict(index,heap[0],top)

    index.pop(top)
    heap.pop(-1)
    sink(heap,index,0)

    return top

def replace(array,index,obj1,obj2):
     id3=-1
     if obj1 in index.keys():
         id=index[obj1]
         array[id]=obj2
         index.pop(obj1)
         index[obj2]=id
         id2=sift(array,index,id)
         id3=sink(array,index,id2)
     else:
         id3=push(array,index,obj2)

     return id3





