# mapreduce_framework.py: 분산 처리 프레임워크
# MapReduce 기반 분산 처리 구현
# Fragment-and-Replicate Join 지원
# 버킷 정렬 구현
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Iterable
from multiprocessing import Pool
import numpy as np
from pathlib import Path
import pickle

class MapReduceFramework:
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.pool = Pool(n_workers)
    
    def fragment_data(self, data: np.ndarray, n_fragments: int) -> List[np.ndarray]:
        """데이터를 n_fragments 개로 분할"""
        chunk_size = len(data) // n_fragments
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    def map_phase(self, mapper: 'Mapper', data_fragments: List[np.ndarray]) -> List[Dict]:
        """Map 단계: 각 워커가 데이터 조각을 독립적으로 처리"""
        return self.pool.starmap(mapper.map_func, [(frag,) for frag in data_fragments])
    
    def shuffle_phase(self, mapped_results: List[Dict]) -> Dict[Any, List]:
        """Shuffle 단계: 키별로 결과 그룹화"""
        shuffled = {}
        for result in mapped_results:
            for key, values in result.items():
                if key not in shuffled:
                    shuffled[key] = []
                shuffled[key].extend(values)
        return shuffled
    
    def reduce_phase(self, reducer: 'Reducer', shuffled: Dict[Any, List]) -> Dict[Any, Any]:
        """Reduce 단계: 각 키별로 집계"""
        return {
            key: reducer.reduce_func(values)
            for key, values in shuffled.items()
        }

class Mapper(ABC):
    @abstractmethod
    def map_func(self, data: np.ndarray) -> Dict[Any, List]:
        """데이터 조각을 처리하여 (key, value) 쌍의 딕셔너리 반환"""
        pass

class Reducer(ABC):
    @abstractmethod
    def reduce_func(self, values: List) -> Any:
        """키에 해당하는 값들을 집계"""
        pass

class FragmentedJoin:
    """Fragment-and-Replicate Join 구현"""
    def __init__(self, n_fragments: int = 4):
        self.n_fragments = n_fragments
    
    def fragment_and_replicate(
        self, 
        left: np.ndarray,
        right: np.ndarray,
        left_key_func: callable,
        right_key_func: callable
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        left와 right 데이터셋을 fragment하고 replicate하여 조인 준비
        - left: 큰 데이터셋 (fragment)
        - right: 작은 데이터셋 (replicate)
        """
        # 1. Fragment left data
        left_fragments = np.array_split(left, self.n_fragments)
        
        # 2. Create hash buckets for right data
        right_buckets = {}
        for r in right:
            key = right_key_func(r)
            bucket_idx = hash(key) % self.n_fragments
            if bucket_idx not in right_buckets:
                right_buckets[bucket_idx] = []
            right_buckets[bucket_idx].append(r)
        
        # 3. Prepare fragment pairs
        pairs = []
        for i, left_frag in enumerate(left_fragments):
            right_frag = np.array(right_buckets.get(i, []))
            if len(right_frag) > 0:
                pairs.append((left_frag, right_frag))
        
        return pairs

class BucketSort:
    """버킷 소팅 구현"""
    def __init__(self, n_buckets: int = 10):
        self.n_buckets = n_buckets
    
    def sort(self, arr: List[Tuple[float, Any]]) -> List[Tuple[float, Any]]:
        """
        (score, value) 튜플 리스트를 버킷 소팅으로 정렬
        - 점수 기반 정렬에 최적화
        """
        # 1. Find range
        if not arr:
            return arr
        min_val = min(x[0] for x in arr)
        max_val = max(x[0] for x in arr)
        if max_val == min_val:
            return sorted(arr, reverse=True)
        
        # 2. Create buckets
        buckets = [[] for _ in range(self.n_buckets)]
        for score, value in arr:
            # Scale to [0, n_buckets-1]
            idx = int((score - min_val) * (self.n_buckets - 1) / (max_val - min_val))
            buckets[idx].append((score, value))
        
        # 3. Sort individual buckets
        for bucket in buckets:
            bucket.sort(reverse=True)  # 내림차순 정렬
        
        # 4. Concatenate buckets
        result = []
        for bucket in reversed(buckets):  # 높은 점수부터
            result.extend(bucket)
        
        return result

def save_fragments(fragments: List[np.ndarray], base_path: str):
    """프래그먼트를 디스크에 저장"""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for i, fragment in enumerate(fragments):
        with open(base_path / f"fragment_{i}.pkl", "wb") as f:
            pickle.dump(fragment, f)

def load_fragments(base_path: str) -> List[np.ndarray]:
    """디스크에서 프래그먼트 로드"""
    base_path = Path(base_path)
    fragments = []
    
    for path in sorted(base_path.glob("fragment_*.pkl")):
        with open(path, "rb") as f:
            fragments.append(pickle.load(f))
    
    return fragments 