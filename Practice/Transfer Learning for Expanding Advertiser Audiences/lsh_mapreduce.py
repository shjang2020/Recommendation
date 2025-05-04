# lsh_mapreduce.py: LSH 분산 처리
# 분산 환경에서의 LSH 구현
# 효율적인 이웃 검색 처리
from typing import Dict, List, Any
import numpy as np
from annoy import AnnoyIndex
from mapreduce_framework import Mapper, Reducer

class LSHMapper(Mapper):
    def __init__(self, dim: int, n_trees: int, metric: str = 'angular'):
        self.dim = dim
        self.n_trees = n_trees
        self.metric = metric
    
    def map_func(self, data: np.ndarray) -> Dict[Any, List]:
        """
        데이터 프래그먼트에 대해 LSH 인덱스 생성 및 이웃 검색
        Returns: {seed_id: [(neighbor_id, distance)]}
        """
        # 1. Build LSH index
        index = AnnoyIndex(self.dim, self.metric)
        for i, vec in enumerate(data):
            index.add_item(i, vec)
        index.build(self.n_trees)
        
        # 2. Find neighbors for each vector
        results = {}
        for i, vec in enumerate(data):
            neighbors = index.get_nns_by_vector(vec, 100, include_distances=True)  # top 100
            results[i] = list(zip(*neighbors))  # (indices, distances) → [(idx, dist),...]
        
        return results

class LSHReducer(Reducer):
    def __init__(self, top_k: int = 200):
        self.top_k = top_k
    
    def reduce_func(self, values: List) -> List:
        """
        여러 LSH 인덱스에서 찾은 이웃들을 병합하고 top-k 선택
        """
        # 1. Count occurrences and aggregate distances
        neighbor_stats = {}
        for neighbors in values:
            for idx, dist in neighbors:
                if idx not in neighbor_stats:
                    neighbor_stats[idx] = {"count": 0, "total_dist": 0}
                neighbor_stats[idx]["count"] += 1
                neighbor_stats[idx]["total_dist"] += dist
        
        # 2. Calculate average distance and sort
        neighbors = [
            (
                idx,
                stats["count"],
                stats["total_dist"] / stats["count"]
            )
            for idx, stats in neighbor_stats.items()
        ]
        
        # 3. Sort by count (desc) and avg distance (asc)
        neighbors.sort(key=lambda x: (-x[1], x[2]))
        
        # 4. Return top-k neighbor indices
        return [n[0] for n in neighbors[:self.top_k]]

class LSHProcessor:
    def __init__(self, n_workers: int = 4, n_trees: int = 10, top_k: int = 200):
        self.n_workers = n_workers
        self.n_trees = n_trees
        self.top_k = top_k
    
    def process(self, embeddings: np.ndarray, seed_indices: List[int]) -> Dict[int, List[int]]:
        """
        MapReduce를 사용하여 LSH 기반 이웃 검색 수행
        """
        from mapreduce_framework import MapReduceFramework
        
        # 1. Initialize framework
        framework = MapReduceFramework(n_workers=self.n_workers)
        
        # 2. Fragment data
        fragments = framework.fragment_data(embeddings, self.n_workers)
        
        # 3. Initialize mapper and reducer
        mapper = LSHMapper(dim=embeddings.shape[1], n_trees=self.n_trees)
        reducer = LSHReducer(top_k=self.top_k)
        
        # 4. Run MapReduce
        mapped = framework.map_phase(mapper, fragments)
        shuffled = framework.shuffle_phase(mapped)
        reduced = framework.reduce_phase(reducer, shuffled)
        
        # 5. Extract results for seed indices
        return {
            seed_idx: reduced[seed_idx]
            for seed_idx in seed_indices
        } 