# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       sasuke
   date：          2019/11/13
-------------------------------------------------
   Change Activity:
                   2019/11/13:
-------------------------------------------------
"""
__author__ = 'sasuke'
import numpy as np
import time
import scipy.io as scio
from scipy import sparse

def matmul_three_mat(A1,A2,A3):
    return np.matmul(np.matmul(A1,A2),A3)


## 方法
def SPUAL(A1, A2, alpha, maxiter, tol, s1, s2, H=None,N1 = None,N2 = None):
    """

    :param A1: 网络1的邻接矩阵
    :param A2: 网络2的邻接矩阵
    :param maxiter:迭代次数
    :param tol: 阈值
    :param s1，s2，alpha: 超参数
    :param H: 初始对齐矩阵
    :param N1: 网络1的属性矩阵
    :param N2:网络1的属性矩阵
    :return:
    """

    print("method myself_2(Final+N)")
    n1 = np.shape(A1)[0]
    n2 = np.shape(A2)[0]
    K = np.shape(N1)[1]
    d1 = np.matmul(A1, np.ones(n1))
    d2 = np.matmul(A2, np.ones(n2))
    d_1 = np.kron(d1, d2)
    N = np.zeros(shape=(n1*n2))
    for i in range(K):
        N = N +np.kron(N1[:, i], N2[:, i])
    d = np.zeros(shape=(n1 * n2))
    for j in range(K):
        d = d+np.kron(np.matmul(A1,N1[:,j]),np.matmul(A2,N2[:,j]))
    D = np.multiply(N, d)
    d = s1*d_1+s2*D
    D = np.power(d, -0.5)
    D[D == np.inf] = 0
    q_1 = D
    q_2 = np.multiply(D, N)
    h = np.reshape(H, newshape=(n2 * n1), order="F")
    s = h
    for iter in range(maxiter):
        prev = s
        M_1 = np.reshape(np.multiply(q_1, s), newshape=(n2, n1), order="F")
        M_2 = np.reshape(np.multiply(q_2, s), newshape=(n2, n1), order="F")
        S_1 = matmul_three_mat(A2, M_1, A1)
        S_2 = matmul_three_mat(A2, M_2, A1)
        S_1 = np.reshape(S_1, newshape=(n2 * n1), order='F')
        S_2 = np.reshape(S_2, newshape=(n2 * n1), order='F')
        a = s1*(np.multiply(q_1,S_1)) + s2*np.multiply(q_2,S_2)
        s = (1 - alpha) * h + alpha * a
        # s = (1 - alpha) * h + alpha * (np.multiply(q, np.reshape(S, newshape=(n2 * n1), order='F')))
        # alpha * np.multiply(np.power(node_attr_sim,2),s)
        delta = np.linalg.norm(s - prev, ord=2)
        print("iteration % d with loss=%f" % (iter, 100 * delta))
        if delta < tol:
            break
    S = np.reshape(s, newshape=(n2, n1), order="F")

    return S

## 贪婪匹配
def match(S):
    m,n = np.shape(S)
    N = m*n
    s = np.reshape(S, newshape=(N), order='F')

    minsize = min(m,n)
    usedrows = np.zeros(shape=(m))
    usedcols = np.zeros(shape=(n))

    maxlist = np.zeros(shape=(minsize))
    row = np.zeros(shape=(minsize))
    col = np.zeros(shape=(minsize))

    ix = np.argsort(s)[::-1]

    y = np.sort(s)[::-1]
    matched = 0
    index = 0
    while matched < minsize:
        ipos = ix[index]+1
        jc = int(np.ceil(ipos/float(m)))
        ic = int(ipos -(jc-1)*m)
        if ic == 0:
            ic= 1
        if usedrows[ic-1] !=1 and usedcols[jc-1] !=1:
            # print(type(matched),matched)
            row[matched] = ic-1
            col[matched] = jc-1
            maxlist[matched] = s[index]
            usedrows[ic-1] = 1
            usedcols[jc-1] = 1
            matched = matched+1
        index  = index + 1
    data = np.ones(shape=(minsize))
   # M = sparse.csr_matrix(row, col, data, m, n)
    M = sparse.csr_matrix((data, (row, col)), shape=(m, n))
    return M


##计算acc
def score_acc(M,true_alignments,gndtruth_num):
    row, col = np.nonzero(M)
    row_true, col_true = np.nonzero(true_alignments)
    index = list(zip(row, col))
    index_true = list(zip(row_true, col_true))
    score = 0
    for i in range(len(index)):
        row_ = index[i][0]
        col_ = index[i][1]
        if true_alignments[row_][col_] == 1:
            score += 1
    return  score / float(gndtruth_num)


## 加载数据
def load_data(file_path,str1,str2):
    """
    :param file_path:数据位置
    :param str1: 网络1名称
    :param str2: 网络2名称
    :return: 数据
    """
    data = scio.loadmat(file_path)
    A1 = data[str1].toarray()
    A2 = data[str2].toarray()
    N1 = data[str1+"_node_label"].toarray()
    N2 = data[str2+"_node_label"].toarray()
    E1 = data[str1 + "_edge_label"]
    E2 = data[str2 + "_edge_label"]
    H = data["H"].toarray()
    gndtruth = data["gndtruth"]
    gndtruth_num = np.shape(gndtruth)[0]
    true_alignments = np.zeros(shape=(np.shape(A2)[0], np.shape(A1)[0]))  ##真正的对齐矩阵
    true_alignments_dict = {} ## 真正的对齐字典
    for i in range(gndtruth_num):
        true_alignments[int(gndtruth[i][1]-1),int(gndtruth[i][0]-1)] = 1
        true_alignments_dict[int(gndtruth[i][1]-1)] = int(gndtruth[i][0]-1)
    return A1,A2,N1,N2,H,true_alignments,gndtruth_num,true_alignments_dict, E1,E2

## 计算当前用户最相似的用户
def top_k(align_mat,user,k,network):
    """
    :param align_mat: 对齐矩阵
    :param user: 待对齐的用户
    :param k: top——k
    :param network:  待对齐的用户属于哪个网络
    :return: k个最相似的用户
    """
    network1_num = len(align_mat)
    network2_num = len(align_mat[0])
    if network == 0:
        if user not in range(network1_num):
            print("此用户不存在")
            return
        scores = align_mat[user]
    else:
        if user not in range(network2_num):
            print("此用户不存在")
            return
        scores = align_mat[:,user]

    length = len(scores)
    scores_dic = {}
    for i in range(length):
        scores_dic[i] = scores[i]
    result = sorted(scores_dic.items(),key=lambda item:item[1],reverse=True)
    return result[:k]


if __name__ == "__main__":
    file_path = "../data/flickr-myspace.mat"
    A1, A2, N1, N2, H, true_alignments, gndtruth_num, true_alignments_dict, E1, E2 = load_data(file_path, "flickr", "myspace")
    alpha = 0.3
    maxiter = 30
    tol = 1e-4
    s1 = 0.7
    s2 = 0.3
    ## 计算对齐矩阵
    S = SPUAL(A1=A1, A2=A2, alpha=alpha, maxiter=maxiter, tol=tol, H=H, N1=N1, N2=N2, s1=s1, s2=s2)
    np.save("../date/align_mat",S)
    ## 计算acc
    M = match(S)
    score = score_acc(M, true_alignments, gndtruth_num)
    print("acc_{}_{}".format(s1, s2), score)
    cur_user =  50
    k = 3
    top_k_user = top_k(align_mat=S,user = cur_user,k=k,network=0)
    print(top_k_user)


