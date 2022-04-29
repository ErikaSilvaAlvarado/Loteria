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
    L= len(index)
    aS = np.empty(L,dtype=int)
    for i in range(L):
        k = index[i]
        aS[i]=array[k]
    return aS



def NearestLower(A, low, high, key):
    if high < low:
        return -1
    i = math.floor((high-low)/2 + low)
    aux = A[i]
    if key == A[i]:
        return i
    elif key < A[i]:
        high = i-1      #search restricted to the lower half of A
        return NearestLower(A,low,high,key)
    else: #key > A[i]
        if A[i+1]>key:
            return i
        else:
            low = i+1       #search restricted to the upper half of A
            return NearestLower(A, low, high, key)

def SelectSets(A, B, p):
    # Sorting points array in non-decreasing order with their original indices
    ps, ip = MergeSortPlusIndex(p, np.array(list(range(M))))
    # Sorting lower limit array in non-decreasing order
    As, ia = MergeSortPlusIndex(A, np.array(list(range(len(A)))))
    # Sorting upper limit array in non-decreasing order
    Bs, ib = MergeSortPlusIndex(B, np.array(list(range(len(B)))))
    # Discard sets closing before p[0]
    posb = NearestLower(Bs, 0, N, p[0])
    B = Bs[posb + 1:]
    ib = ib[posb + 1:]
    # A,B are sorted by B non decremental
    A = SortArray(A, ib)
    As, ia2 = MergeSortPlusIndex(A, np.array(list(range(len(A)))))
    # Discard sets opening after maxPs
    posa = NearestLower(As, 0, N, ps[- 1])
    A = As[0:posa + 1]
    ia = ia2[0:posa + 1]
    # A,B are sorted by A non decremental
    B = SortArray(B, ia)
    return A,B

def LotteryFunction(A, B, p):
    Na = len(A)
    Nb = len(B)
    N = max(Na,Nb)
    M = len(p)
    if N == 1:  # only one segment
        lottery = np.zeros(M, dtype=int)
        # Sorting points array in non-decreasing order with their original indices
        ps, ip = MergeSortPlusIndex(p, np.array(list(range(M))))
        # min index points afer A
        lp = NearestLower(ps, 0, M, A[0])+1
        # max index points before B
        rp = NearestLower(ps, 0, M, B[0])
        for j in range (lp, rp+1):
            k = ip[j]
            lottery[k] = 1
        return True, lottery
    elif M==1: #one single point
        lottery = 0
        #Sort and select sets
        A,B = SelectSets(A, B, p)
        lottery = N
        print(lottery)
        return True, lottery
    else:
        # Sort and select sets
        A, B = SelectSets(A, B, p)
        #part B in two halfs
        halfN = math.floor((N + 1) / 2)
        B1 = B[0:halfN]
        B2 = B[halfN:]
        # Sort points array in non-decreasing order with their original indices
        ps, ip = MergeSortPlusIndex(p, np.array(list(range(M))))
        #send points according to the B partitions
        indexP = NearestLower(p, 0, M - 1, B1[-1])
        p1 = ps[0:indexP + 1]
        p2 = ps[indexP + 1:M]
        # Sort A according to the B partitions
        indexA = NearestLower(A, 0, N - 1, B1[-1])
        A1 = A[0:indexA + 1]
        A2 = A[indexA + 1:]
        flag1, lottery1 = LotteryFunction(A1, B1, p1)
        flag2, lottery2 = LotteryFunction(A2, B2, p2)
        if flag1 == True:
                if len(A1) > 0:
                    for j in range(len(p1)):
                        i = 0
                        while i < len(A1) and A1[i] <= p1[j]:
                            count[j] += 1
                            i += 1
                        k = ip[lp+j]
                        lottery[k] = count[j]
        return True, lottery


if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    #input variables
    N = data[0]                 # number of segments
    M = data[1]                 # number of values to find
    A = np.array(data[2:2 * N + 2:2])    # lower limit of each segment
    B = np.array(data[3:2 * N + 2:2])    # upper limit of each segment
    p= np.array(data[2 * N + 2:])   # points to find
    flag,results = LotteryFunction(A, B, p)

