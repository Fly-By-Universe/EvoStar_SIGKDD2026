from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
try:
    from memory_profiler import memory_usage
except Exception:
    memory_usage = None
import os
import time
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from sklearn.preprocessing import MinMaxScaler
from enum import Enum, unique
from functools import wraps
import heapq

from numba import jit

query_budget = 5500000
oracle_cache = {}


@jit(nopython=True, fastmath=True)
def get_upper_tri_distances(dist_matrix):
    n = dist_matrix.shape[0]
    count = n * (n - 1) // 2
    res = np.empty(count, dtype=dist_matrix.dtype)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            res[idx] = dist_matrix[i, j]
            idx += 1
    return res


@jit(nopython=True, fastmath=True)
def calc_densities_and_find_best(dist_matrix, cutoff, degrees):
    n = dist_matrix.shape[0]
    # Calculate densities manually to avoid allocating boolean mask matrix
    densities = np.zeros(n, dtype=np.int32)
    for i in range(n):
        d = 0
        for j in range(n):
            if dist_matrix[i, j] <= cutoff:
                d += 1
        densities[i] = d - 1  # Subtract self

    # Find max density
    max_d = -1
    for i in range(n):
        if densities[i] > max_d:
            max_d = densities[i]

    # Find best node (tie break with degree)
    best_idx = -1
    best_degree = -1

    for i in range(n):
        if densities[i] == max_d:
            if best_idx == -1:
                best_idx = i
                best_degree = degrees[i]
            else:
                if degrees[i] > best_degree:
                    best_idx = i
                    best_degree = degrees[i]
    return best_idx


GLOBAL_DATA = None


def save_metric_to_csv(dataName, counts, values, metric_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({'': counts, metric_name: values})
    file_path = os.path.join(save_dir, f"{dataName}_result.csv")
    df.to_csv(file_path, index=False)


def data_processing(filePath: str,
                    minMaxScaler=True,
                    drop_duplicates=True,
                    shuffle=False):
    data = pd.read_csv(filePath, header=None)
    for col in data.columns[:-1]:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    true_k = data.iloc[:, -1].drop_duplicates().shape[0]
    if drop_duplicates:
        data = data.drop_duplicates().reset_index(drop=True)
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    if minMaxScaler:
        try:
            numeric_columns = data.columns[:-1]
            for col in numeric_columns:
                if data[col].dtype in ['int64', 'int32', 'float32', 'object']:
                    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float32')
            data.iloc[:, :-1] = MinMaxScaler().fit_transform(data.iloc[:, :-1])
        except Exception as e:
            print(f"数据类型转换出错: {e}")
            data.iloc[:, :-1] = MinMaxScaler().fit_transform(data.iloc[:, :-1])
    return np.array(data.iloc[:, :-1]), data.iloc[:, -1].values.tolist(), true_k


def print_estimate(label_true, label_pred, dataset: str, testIndex, iterIndex, time, params='无'):
    print('%15s 数据集，参数%10s，第 %2d次测试，第 %2d次聚类，真实簇 %4d个，预测簇 %4d个，用时 %10.2f s,'
          'RI %0.4f,ARI %0.4f,NMI %0.4f，'
          % (dataset.ljust(15)[:15], params.ljust(10)[:10], int(testIndex), int(iterIndex),
             len(list(set(label_true))), len(list(set(label_pred))), time,
             rand_score(label_true, label_pred),
             adjusted_rand_score(label_true, label_pred), 
             normalized_mutual_info_score(label_true, label_pred)))


class Node(object):
    id: int
    data: tuple
    label: int
    labelName: str
    adjacentNode: dict
    degree: int
    iteration: int
    isVisited: bool
    label_pred: int
    parent: Node
    query: bool
    children: int
    root_judge: list

    def __init__(self, node_id, data, label, labelName):
        self.id = int(node_id)
        self.data = data
        self.label = label
        self.labelName = labelName
        self.adjacentNode = {}
        self.degree = 0
        self.iteration = 0
        self.isVisited = False
        self.node_num = 0
        self.parent = None
        self.children = 0
        self.label_pred = 0
        self.root_judge = []
        self.query = False

    def add_adjacent_node(self, node: Node):
        self.adjacentNode[node.id] = node
        self.degree += 1

    def set_iteration(self, iteration: int):
        self.iteration = iteration

    def set_node_num(self, node_num: int):
        self.node_num = node_num


class Graph(object):
    nodeList: list[Node]
    node_size: int

    def __init__(self):
        self.nodeList = []
        self.node_size = len(self.nodeList)

    def add_Node(self, node: Node):
        self.nodeList.append(node)
        self.node_size = len(self.nodeList)


class ClusterResult:
    def __init__(self, dataName, iteration, roots, execution_time, ri, ari, nmi):
        self.dataName = dataName
        self.iteration = iteration
        self.roots = roots
        self.execution_time = execution_time
        self.ri = ri
        self.ari = ari
        self.nmi = nmi

    def __str__(self):
        return (f"ClusterResult(dataName={self.dataName}, iteration={self.iteration}, "
                f"execution_time={self.execution_time:.2f}s, ri={self.ri:.4f}, ari={self.ari:.4f}, "
                f"nmi={self.nmi:.4f})")


class ClusterStoreWithHeap:
    def __init__(self):
        self.heap = []

    def add_result(self, cluster_result: ClusterResult):
        heapq.heappush(self.heap, (cluster_result.execution_time, cluster_result))

    def get_best_result(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        return None

    def print_results(self):
        for _, result in self.heap:
            print(result)
            for root_id in result.roots:
                pass

    def __str__(self):
        return


def get_distribute(current_roots: list[int], nodeList: dict[int, Node], size: int):
    label_true = [-1 for i in range(size)]
    label_pred = [-1 for i in range(size)]
    i = 0
    count = 0
    for root_id in current_roots:
        node = nodeList[root_id]
        next_node = [node]
        visited = {node.id}
        while next_node:
            r = next_node.pop()
            label_pred[r.id] = i
            label_true[r.id] = r.label
            count += 1
            for n in r.adjacentNode:
                if n not in visited:
                    visited.add(n)
                    next_node.append(r.adjacentNode[n])
        i += 1
    return label_true, label_pred, count


def extract_pairs_distance(roots, iteration):
    scts = []
    for key in roots:
        sct_node = [roots[key]]
        sct_indices = [roots[key].id]
        node_num = 1
        next_node = [roots[key]]
        visited = {key}
        while next_node:
            r = next_node.pop()
            for n in r.adjacentNode:
                if r.adjacentNode[n].iteration == iteration and n not in visited:
                    visited.add(n)
                    next_node.append(r.adjacentNode[n])
                    sct_node.append(r.adjacentNode[n])
                    sct_indices.append(r.adjacentNode[n].id)
                    node_num += 1
        for node in sct_node:
            node.set_node_num(node_num)
        scts.append(dict(sct_indices=sct_indices, sct_node=sct_node))
    return scts


def format_distance(distance):
    # Optimized using Numba
    d = get_upper_tri_distances(distance)
    d.sort()
    return d


def compute_local_density(nodeList, distance, cut_off_distance):
    # Optimized using Numba
    degrees = np.array([node.degree for node in nodeList], dtype=np.int32)
    best_idx = calc_densities_and_find_best(distance, cut_off_distance, degrees)
    return nodeList[int(best_idx)]


def compute_degree_weigh(point):
    repeated_point = [p for p in point if p['local_density_node_num'] == point[0]['local_density_node_num']]
    d = repeated_point[0]['node']
    if len(repeated_point) != 1:
        base_degree = repeated_point[0]['node'].degree
        for p in repeated_point:
            if p['node'].degree > base_degree:
                d = p['node']
                base_degree = p['node'].degree
    return d


def findDensityPeak(query_times: int, roots: list[Node], cut_off=0.4, iteration=0):
    scts = extract_pairs_distance(roots, iteration)
    rootList = {}
    for sct in scts:
        if len(sct['sct_indices']) > 1:
            sct_indices = sct['sct_indices']
            sct_data = GLOBAL_DATA[sct_indices]
            distances = pairwise_distances(sct_data, metric="euclidean")
            pairs_distance = format_distance(distances)
            if pairs_distance.size == 0:
                cut_off_distance = 0.0
            else:
                index = min(int(round(pairs_distance.size * cut_off)), pairs_distance.size - 1)
                cut_off_distance = float(pairs_distance[index])
            
            root = compute_local_density(sct['sct_node'], distances, cut_off_distance)
            
            rootList[root.id] = root
            for node in sct['sct_node']:
                node.parent = root
                node.root_judge.append(0)
            for node in sct['sct_node']:
                if node.id != root.id:
                    to_remove = []
                    for adj_id in node.adjacentNode:
                        adj = node.adjacentNode[adj_id]
                        if adj.iteration == iteration and adj.id != root.id:
                            to_remove.append(adj_id)
                    for adj_id in to_remove:
                        if adj_id in node.adjacentNode:
                            del node.adjacentNode[adj_id]
                            node.degree -= 1
                    if root.id not in node.adjacentNode:
                        node.add_adjacent_node(root)
                    if node.id not in root.adjacentNode:
                        root.add_adjacent_node(node)
            root.root_judge[-1] = 1
        else:
            root = sct['sct_node'][0]
            rootList[root.id] = root
            root.parent = root
            root.root_judge.append(1)
    for sct in scts:
        for node in sct['sct_node']:
            if node.id not in rootList and node.parent and node.parent.id in rootList:
                continue
            elif node.id not in rootList:
                rootList[node.id] = node
                node.parent = node
                node.root_judge.append(1)
    return rootList


def extract_data_from_Node(nodeList: dict[int, Node]):
    return [nodeList[key].data for key in nodeList]


def findNNs(query_times: int, nodeList: list[Node], k):
    dataList = extract_data_from_Node(nodeList)
    return kdTree(query_times, dataList, nodeList, k, return_dist=False)


def kdTree(query_times: int, dataList, nodeList: dict[int, Node], k, return_dist=False):
    origin = np.asarray(dataList)
    k = min(k, len(dataList))
    neighbors = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(origin)
    indices = neighbors.kneighbors(origin, return_distance=False)
    nns = {}
    snns = {}
    pos = [key for key in nodeList]
    for i, key in enumerate(nodeList):
        if k > 2:
            nns[nodeList[key].id] = pos[indices[i][1]]
            snns[nodeList[key].id] = pos[indices[i][2]]
        elif k == 2:
            nns[nodeList[key].id] = pos[indices[i][1]]
            snns[nodeList[key].id] = pos[indices[i][1]]
        else:
            nns[nodeList[key].id] = nodeList[key].id
            snns[nodeList[key].id] = nodeList[key].id
    return nns, snns


def compute_sct_num(roots: dict[int, Node], iteration: int):
    rebuild_roots = []
    for key in roots:
        sct_node = [roots[key]]
        node_num = 1
        next_node = [roots[key]]
        other_node = 0
        visited = {key}
        while next_node:
            r = next_node.pop()
            for n in r.adjacentNode:
                if r.adjacentNode[n].iteration == iteration and n not in visited:
                    next_node.append(r.adjacentNode[n])
                    sct_node.append(r.adjacentNode[n])
                    visited.add(n)
                    other_node = n
                    node_num += 1
        if node_num == 2:
            rebuild_roots.append((key, other_node))
        for node in sct_node:
            node.set_node_num(node_num)
    return rebuild_roots


def construction(nodeList: list[Node], nns: dict[int], snns: dict[int], iteration: int, query_times: int):
    nodeDict = {node.id: node for node in nodeList}
    roots = {}
    
    if iteration > 0:
        sorted_candidates = sorted(nodeList, key=lambda x: x.degree, reverse=True)
    else:
        sorted_candidates = list(nodeList) # Default order

    visited = set()
    candidate_ids = set(node.id for node in nodeList)
    
    ptr = 0
    num_nodes = len(sorted_candidates)
    min_clusters_required = max(1, int(len(nodeList) / 1000000))

    while ptr < num_nodes:
        # Fast-forward to next unvisited node
        start_node = None
        while ptr < num_nodes:
            node = sorted_candidates[ptr]
            if node.id not in visited:
                start_node = node
                break
            ptr += 1
        
        if start_node is None:
            break
            

        if (len(candidate_ids) + len(roots)) <= min_clusters_required:
             break

        current_node = start_node
        link = []
        link_set = set()
        
        while True:
            if current_node.id in visited and current_node.id not in link_set:
                if len(link) > 0:
                    root_node = nodeDict[link[-1]]
                    roots[root_node.id] = root_node
                else:
                    pass
                break          
            if current_node.id in link_set:
                pass
            
            if current_node.id in visited and current_node.id not in link_set:
                pass

            visited.add(current_node.id)
            candidate_ids.discard(current_node.id)
            link.append(current_node.id)
            link_set.add(current_node.id)
            
            current_node.set_iteration(iteration)
            
            j_id = nns[current_node.id]
            j = nodeDict[j_id]
            
            # Case 1: Cycle in current path
            if j_id in link_set:
                roots[current_node.id] = current_node
                break
                
            # Case 2: Hit a node already processed in a previous tree (not in candidates)
            elif j_id not in candidate_ids:
                current_node.add_adjacent_node(j)
                j.add_adjacent_node(current_node)
                break
                
            # Case 3: Continue growing
            else:
                current_node.add_adjacent_node(j)
                j.add_adjacent_node(current_node)
                current_node = j

    # Add remaining candidates as roots
    for node in sorted_candidates:
        if node.id not in visited:
            roots[node.id] = node
            node.set_iteration(iteration)
            
    return roots


def connect_roots(rebuild_roots, roots, snns, nodeList: list[Node], iteration: int, query_times: int):
    if query_times == 0:
        nodeDict = {node.id: node for node in nodeList}
        candidates = np.array(rebuild_roots).reshape(-1)
        for root in rebuild_roots:
            root_node_0 = nodeDict[root[0]]
            root_node_1 = nodeDict[root[1]]
            roots.pop(root[0])
            left_connect_node = nodeDict[snns[root[0]]].node_num
            right_connect_node = nodeDict[snns[root[1]]].node_num
            if left_connect_node <= right_connect_node:
                big_node = nodeDict[snns[root[1]]]
                small_node = nodeDict[root[1]]
            else:
                big_node = nodeDict[snns[root[0]]]
                small_node = nodeDict[root[0]]
            big_node.add_adjacent_node(small_node)
            small_node.add_adjacent_node(big_node)
            if small_node.id in candidates:
                roots[small_node.id] = small_node


def rebuild(snns: dict[int], roots: list[Node], nodeList: list[Node], iteration: int, query_times):
    rebuild_roots = compute_sct_num(roots, iteration)
    connect_roots(rebuild_roots, roots, snns, nodeList, iteration, query_times)


class Task():
    def __init__(self, params, iterIndex: int, dataName: str, path):
        self.params = params
        self.iterIndex = str(iterIndex)
        self.dataName = str(dataName)
        self.filePath = str(path)

    def __str__(self):
        return '{}-{}'.format(self.dataName, self.iterIndex)


@unique
class RecordType(Enum):
    assignment = 'assignment'
    tree = 'tree'


class Assignment():
    def __init__(self, types: str, iter: str, record: dict):
        self.type = types
        self.iter = iter
        self.record = record


class Record():
    def __init__(self):
        self.record = []
        self.cpuTime = []

    def save_output(self, types: RecordType, label_true: list, label_pred: list, iter=0):
        assert isinstance(types, RecordType), TypeError(
            '输入类型必须为RecordType枚举类，请检查。当前类型为 ' + str(type(types)))
        assert len(label_pred) > 0, \
            TypeError('label_pred必须为list类型，且长度不为0')
        assert len(label_pred) > 0, \
            TypeError('label_true必须为list类型，且长度不为0')
        self.record.append(
            Assignment(types, str(iter), {'label_true': label_true, 'label_pred': label_pred}))

    def save_time(self, cpuTime):
        assert isinstance(cpuTime, float), TypeError("输入类型必须为float类型，请检查。当前类型为 " + str(type(cpuTime)))
        self.cpuTime.append(cpuTime)


class ExpMonitor():
    def __init__(self, expId: str, algorithmName: str, storgePath="C:\\Users\\DELL\\OneDrive\\桌面\\dezh123"):
        self.task = None
        self.expId = expId
        self.algorithmName = algorithmName
        self.storgePath = storgePath
        self.stop_thread = False

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.task = kwargs['task'] if 'task' in kwargs else args[0]
            res = func(*args, **kwargs)
            return res
        return wrapper

    def stop_monitor_thread(self):
        self.stop_thread = True

    def lineOutput(self, path, fileName, data: dict):
        self.makeDir(path)
        outputPath = os.path.join(path, fileName + '.csv')
        if not os.path.exists(outputPath):
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, mode='a')
        else:
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, header=False, mode='a')

    def makeDir(self, path):
        os.makedirs(path, exist_ok=True)


def query_oracle(node1, node2):
    id1, id2 = node1.id, node2.id
    if id1 > id2:
        id1, id2 = id2, id1
    key = (id1, id2)
    
    cached = oracle_cache.get(key)
    if cached is not None:
        return cached
    
    res = "must_link" if node1.label == node2.label else "cannot_link"
    oracle_cache[key] = res
    return res


def process_uncertain_nodes(nodes, nodeList, current_roots, label, query_times, n_parts=1, eval_every=2):
    ari_values = []
    nmi_values = []
    query_counts = []
    total_nodes = len(nodes)
    parts = []
    for part in range(n_parts):
        start_idx = (total_nodes * part) // n_parts
        end_idx = (total_nodes * (part + 1)) // n_parts
        if start_idx < end_idx:
            parts.append(nodes[start_idx:end_idx])
    for part in parts:
        for node in part:
            query_times, nodeList, current_roots, updated = process_single_node(
                node, nodeList, current_roots, label, query_times)
            if eval_every is not None and eval_every > 0 and query_times % (eval_every) == 0:
                label_true, label_pred, count = get_distribute(current_roots, nodeList, len(label))
                current_ari = adjusted_rand_score(label_true, label_pred)
                current_nmi = normalized_mutual_info_score(label_true, label_pred)
                ari_values.append(current_ari)
                nmi_values.append(current_nmi)
                query_counts.append(query_times)
                if query_times % (eval_every*50) == 0:
                    print("当前查询次数：", query_times, "当前ARI值：", current_ari, "当前NMI值：", current_nmi)
                if current_ari >= 0.99 or query_times >= query_budget:
                    return query_times, nodeList, current_roots, ari_values, nmi_values, query_counts
    return query_times, nodeList, current_roots, ari_values, nmi_values, query_counts


def process_single_node(node, nodeList, current_roots, label, query_times, print_node_info=True):
    updated = False
    if node.parent is None:
        return query_times, nodeList, current_roots, updated
    if node.id != node.parent.id and node.query == False:
        ans = query_oracle(node.parent, node)
        query_times += 1
        updated = True
        if ans == "must_link":
            node.query = True
        elif ans == "cannot_link":
            node.query = True
            flag = False
            parent_center = node.parent
            while parent_center.parent is not None and parent_center.parent.id != parent_center.id:
                parent_center = parent_center.parent
            candidates = [i for i in nodeList.values() if i.id != parent_center.id]
            node_data = np.asarray(node.data, dtype=np.float32)
            if len(candidates) > 0:
                centers = np.asarray([c.data for c in candidates], dtype=np.float32)
                dists = np.linalg.norm(centers - node_data, axis=1)
                order = np.argsort(dists)
                for idx in order:
                    i = candidates[int(idx)]
                    ans_2 = query_oracle(i, node)
                    query_times += 1
                    if ans_2 == "must_link":
                        flag = True
                        true_root = i
                        node.label_pred = i.label_pred
                        break
            old_parent = node.parent
            if node.id in old_parent.adjacentNode:
                del old_parent.adjacentNode[node.id]
                old_parent.degree -= 1
            if old_parent.id in node.adjacentNode:
                del node.adjacentNode[old_parent.id]
                node.degree -= 1
            if flag == False:
                current_roots.append(node.id)
                nodeList[node.id] = node
                node.parent = node
            elif flag == True:
                node.add_adjacent_node(true_root)
                true_root.add_adjacent_node(node)
                node.parent = true_root
    return query_times, nodeList, current_roots, updated


@ExpMonitor(expId='EVOSTAR', algorithmName='EVOSTAR', storgePath="C:\\Users\\DELL\\OneDrive\\桌面\\dezh123")
def run(task: Task, data_override=None, n_parts = 1):
    global query_times
    global GLOBAL_DATA
    query_times = 0
    record = Record()
    ari_values = []
    nmi_values = []
    query_counts = []
    wall_start = time.perf_counter()
    if data_override is not None:
        data, label, K = data_override
    else:
        data, label, K = data_processing(task.filePath)
    
    # Initialize Global Data Matrix for fast access
    # Ensure data is contiguous and float32 for speed
    GLOBAL_DATA = np.ascontiguousarray(data, dtype=np.float32)
    
    n = len(data)
    nodeList = {i: Node(i, data[i], label[i], label[i]) for i in range(len(data))}
    iteration = 0
    algo_start = time.perf_counter()
    iteration_roots_list = []
    ari = 0.0
    nmi = 0.0
    while len(nodeList) > 1:
        iter_start = time.perf_counter()
        nns, snns = findNNs(query_times, nodeList=nodeList, k=3)
        roots = construction(nodeList=list(nodeList.values()), nns=nns, snns=snns, iteration=iteration, query_times=query_times)
        rebuild(snns, roots, list(nodeList.values()), iteration, query_times)
        nodeList = findDensityPeak(query_times, roots, task.params, iteration=iteration)
        current_roots = list(nodeList.keys())
        iteration_roots_list.append(current_roots)
        label_true, label_pred, count = get_distribute(current_roots, nodeList, len(label))
        ari = adjusted_rand_score(label_true, label_pred)
        nmi = normalized_mutual_info_score(label_true, label_pred)
        iter_end = time.perf_counter()
        record.save_time(iter_end - iter_start)
        record.save_output(RecordType.assignment, label_true, label_pred, iteration)
        elapsed_time = iter_end - algo_start
        print_estimate(label_true, label_pred, task.dataName, task.iterIndex, iteration, elapsed_time, params=str(task.params))
        iteration += 1
    ari_values.append(ari)
    nmi_values.append(nmi)
    query_counts.append(query_times)

    def get_all_nodes(roots: dict[int, Node]) -> list[Node]:
        all_nodes = {}
        for root in roots.values():
            stack = [root]
            while stack:
                node = stack.pop()
                if node.id not in all_nodes:
                    all_nodes[node.id] = node
                    for child in node.adjacentNode.values():
                        stack.append(child)
        return list(all_nodes.values())

    all_nodes = get_all_nodes(nodeList)
    top_nodes = get_top_k_uncertain_nodes(all_nodes, top_k=len(data))
    query_times, nodeList, current_roots, higher_ari_values, higher_nmi_values, higher_query_counts = process_uncertain_nodes(
        top_nodes, nodeList, current_roots, label, query_times, n_parts=n_parts
    )
    ari_values.extend(higher_ari_values)
    nmi_values.extend(higher_nmi_values)
    query_counts.extend(higher_query_counts)
    label_true, label_pred, count = get_distribute(current_roots, nodeList, len(label))
    elapsed_time = time.perf_counter() - algo_start
    print_estimate(label_true, label_pred, task.dataName, task.iterIndex, iteration, elapsed_time, params=str(task.params))
    print("\n")
    save_results_to_csv(task.dataName, query_counts, ari_values, nmi_values)
    total_wall = time.perf_counter() - wall_start
    total_algo = elapsed_time
    print(f"{task.dataName} 总用时(算法): {total_algo:.2f}s, 总用时(含数据读取): {total_wall:.2f}s")
    return {'record': record, 'ari_values': ari_values, 'nmi_values': nmi_values, 'query_counts': query_counts}


def save_results_to_csv(dataName, query_counts, ari_values, nmi_values):
    result_dir = "C:\\Users\\DELL\\OneDrive\\桌面\\论文\\WWW"
    os.makedirs(result_dir, exist_ok=True)
    result_df = pd.DataFrame({
        ' ': query_counts,
        'ARI': ari_values,
        'NMI': nmi_values,
    })
    file_path = os.path.join(result_dir, f"{dataName}_result.csv")
    result_df.to_csv(file_path, index=False)
    print(f"结果已保存到: {file_path}")


def get_top_k_uncertain_nodes(nodes, top_k=10000000):
    def get_root_distance(node):
        root = node.parent
        node_data = np.array(node.data)
        root_data = np.array(root.data)
        distance = np.linalg.norm(node_data - root_data)
        return distance

    # Optimized: Use heapq if top_k is significantly smaller than len(nodes), 
    # but here top_k is len(data) (all nodes), so sort is optimal O(N log N).
    nodes_list = list(nodes)
    nodes_list = sorted(nodes_list, key=lambda node: (node.iteration, node.degree, get_root_distance(node)), reverse=True)
    return nodes_list


if __name__ == '__main__':
    run_multi = 0
    single_file_path = r"C:\Users\DELL\OneDrive\桌面\论文\Materials\mnist.csv"
    multi_dir_path = r"D:\ALDP-master\data\datadata"
    cut = 0.19
    if run_multi == 1:
        path = multi_dir_path
        if os.path.isdir(path):
            for dataName in os.listdir(path):
                file_path = os.path.join(path, dataName)
                if not os.path.isfile(file_path):
                    continue
                data, label, K = data_processing(file_path)
                task = Task(round(cut * 1, 2), 1, dataName.split('.')[0], file_path)
                if memory_usage is None:
                    run(task=task)
                else:
                    mem_usage, result = memory_usage((run, (), {'task': task}), retval=True, interval=0.1, max_usage=True)
                    peak_mem = mem_usage if isinstance(mem_usage, float) else max(mem_usage)
                    print(f"{task.dataName} 内存峰值: {peak_mem} MB")
        else:
            raise FileNotFoundError(f"找不到数据集文件夹: {path}")
    else:
        path = single_file_path
        if os.path.isfile(path):
            dataName = os.path.basename(path)
            data, label, K = data_processing(path)
            task = Task(round(cut * 1, 2), 1, dataName.split('.')[0], path)
            if memory_usage is None:
                run(task=task)
            else:
                mem_usage, result = memory_usage((run, (), {'task': task}), retval=True, interval=0.1, max_usage=True)
                peak_mem = mem_usage if isinstance(mem_usage, float) else max(mem_usage)
                print(f"{task.dataName} 内存峰值: {peak_mem} MB")
        else:
            raise FileNotFoundError(f"找不到数据集文件: {path}")

    print("\n")
