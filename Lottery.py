#Uses python3
import sys
import numpy as np
import math

#Asorted, indexSorted = MergeSortPlusIndex(A,index)
def MergeSortPlusIndex(A,index):
    C = np.copy(A)
    N = len(C)
    if N == 1:
        return C, 0
    if N == 2:
        C,  index = TwoElemSortPlusIndex(C,index, 0)
        return C, index
    if N == 3:
        C, index = TwoElemSortPlusIndex(C,index, 0)
        C, index = TwoElemSortPlusIndex(C,index, 1)
        C, index = TwoElemSortPlusIndex(C,index, 0)
        return C,  index
    else:   #N>3:
        halfN = math.floor((N-1)/2)
        #first half
        B1 = C[0: halfN+1]
        index1 = index[0: halfN + 1]
        #second half
        B2 = C[halfN+1:N]
        index2 = index[halfN+1:N]
        #Recursividty
        B1, index1 = MergeSortPlusIndex(B1, index1)
        B2, index2 = MergeSortPlusIndex(B2, index2)
        C, index = MergeArraysPlusIndex(B1, B2, index1, index2)
        return C, index

def TwoElemSortPlusIndex(C,index, i):
    if C[i] > C[i+1]:
        aux = C[i]
        C[i] = C[i+1]
        C[i+1] = aux
        #same process with index array
        aux = index[i]
        index[i] = index[i+1]
        index[i+1] = aux
    return C, index

def MergeArraysPlusIndex(B1,B2, index1, index2):
    N1 = len(B1)
    N2 = len(B2)
    N = N1 + N2
    C = np.empty(N, dtype=int)
    index = np.empty(N, dtype=int)
    i, pos1 , pos2 = 0, 0, 0
    while i < (N-1):
        if B1[pos1] >= B2[pos2]:
            C[i] = B2[pos2]
            index[i] = index2[pos2]
            pos2 += 1
            N2 = N2 - 1
            if N2 != 0:
                i+=1
            else: # N2 == 0
                indices = list(range(i+1, N))
                C.put(indices, B1[pos1:])    #agregar a B1[pos1:]
                index.put(indices, index1[pos1:])
                i = N-1
        else: # B1[pos1] < B2[pos2]
            C[i] = B1[pos1]
            index[i] = index1[pos1]
            N1 = N1 - 1
            if N1 != 0:
                pos1 += 1
                i += 1
            else: # N1 == 0
                indices = list(range(i+1, N))
                C.put(indices, B2[pos2:])
                index.put(indices,index2[pos2:])#agregar a B2[pos2:]
                i = N-1
    return C, index

def SortArray(array, index):
    L= len(array)
    aS = np.empty(L,dtype=int)
    for i in range(L):
        k = index[i]
        aS[i]=array[k]
    return aS

def LotteryFunction(N, M, A, B, points):
    # simplest case
    if M == 0:
        lottery = np.empty(M, dtype=int)
        return lottery
    if N == 1:  # only one segment
        lottery = np.empty(M, dtype=int)
        for j in range(M):
            NOS = 0
            if A[0] <= points[j] and B[0] >= points[j]:
                NOS = 1
            lottery[j] = NOS
        return lottery
    elif M == 1: #only one point
        NOS=0
        for i in range(N):
            if A[i] <= points[0] and B[i] >= points[0]:
                NOS += 1
            lottery = NOS
        return lottery
    else:
        # Sorting points array in non-decreasing order with their original indices
        pS, ipS = MergeSortPlusIndex(points, np.array(list(range(M))))
        # Sorting higher limit array in non-decreasing order
        Bs, iBs = MergeSortPlusIndex(B, np.array(list(range(N))))
        Ab = SortArray(A, iBs)
        lottery = np.empty(M, dtype=int)
        pos = 0
        for j in range(M): # voy por cada punto
            while pos<N and Bs[pos] < pS[j]:#cuento los que cierran antes
                pos = pos+1
            Abs = np.sort(Ab[pos:])
            i = 0
            while i<M-pos and Ab[pos+i] <=pS[j]:
                i=i+1
            k = ipS[j]
            lottery[k] = i
        return lottery

if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    #input variables
    N = data[0]                 # number of segments
    M = data[1]                 # number of values to find
    A = np.array(data[2:2 * N + 2:2])    # lower limit of each segment
    B = np.array(data[3:2 * N + 2:2])    # upper limit of each segment
    points = np.array(data[2 * N + 2:])   # points to find
    results = LotteryFunction(N,M,A,B,points)
    for x in results:
        print(x, end=' ')
