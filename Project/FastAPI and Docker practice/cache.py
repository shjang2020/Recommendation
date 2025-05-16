import json
import hashlib
from typing import Any, Optional
import redis
import os
from logger import app_logger

class CacheManager:
    def __init__(self, host: str = None, port: int = None, db: int = 0):
        """Redis 캐시 매니저 초기화"""
        try:
            # 환경 변수에서 호스트와 포트 가져오기
            host = host or os.getenv('REDIS_HOST', 'redis')  # Docker 서비스 이름으로 변경
            port = port or int(os.getenv('REDIS_PORT', 6379))
            
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True  # 문자열로 디코딩
            )
            # 연결 테스트
            self.redis_client.ping()
            app_logger.info("Redis 연결 성공")
        except redis.ConnectionError as e:
            app_logger.error(f"Redis 연결 실패: {str(e)}")
            self.redis_client = None
            app_logger.warning("Redis 캐시 기능이 비활성화됩니다.")

    def _generate_key(self, genres: Optional[list], keywords: str, num_recommendations: int, start_year: int, end_year: int) -> str:
        """캐시 키 생성"""
        # 입력값을 정렬하여 동일한 입력에 대해 같은 키가 생성되도록 함
        genres_str = ','.join(sorted(genres)) if genres else ''
        keywords_str = keywords if keywords else ''
        
        # 키 생성
        key_parts = [
            'movie_recommendations',
            genres_str,
            keywords_str,
            str(num_recommendations),
            str(start_year),
            str(end_year)
        ]
        
        # 빈 문자열 제거 후 조합
        key = ':'.join(filter(None, key_parts))
        
        # 키가 너무 길 경우 해시 사용
        if len(key) > 100:
            return f"movie_recommendations:{hashlib.md5(key.encode()).hexdigest()}"
        return key

    def get(self, genres: Optional[list], keywords: str, num_recommendations: int, start_year: int, end_year: int) -> Optional[list]:
        """캐시에서 데이터 조회"""
        if not self.redis_client:
            return None

        try:
            key = self._generate_key(genres, keywords, num_recommendations, start_year, end_year)
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                app_logger.info(f"Cache hit for key: {key}")
                return json.loads(cached_data)
            
            app_logger.info(f"Cache miss for key: {key}")
            return None
            
        except Exception as e:
            app_logger.error(f"Cache get error: {str(e)}")
            return None

    def set(self, genres: Optional[list], keywords: str, 
            num_recommendations: int, start_year: int, end_year: int, data: list, expire_time: int = 3600) -> bool:
        """캐시에 데이터 저장"""
        if not self.redis_client:
            return False

        try:
            key = self._generate_key(genres, keywords, num_recommendations, start_year, end_year)
            self.redis_client.setex(
                key,
                expire_time,  # 1시간 후 만료
                json.dumps(data)
            )
            app_logger.info(f"Cache set for key: {key}")
            return True
            
        except Exception as e:
            app_logger.error(f"Cache set error: {str(e)}")
            return False

    def delete(self, genres: Optional[list], keywords: str, num_recommendations: int, start_year: int, end_year: int) -> bool:
        """캐시에서 데이터 삭제"""
        if not self.redis_client:
            return False

        try:
            key = self._generate_key(genres, keywords, num_recommendations, start_year, end_year)
            self.redis_client.delete(key)
            app_logger.info(f"Cache deleted for key: {key}")
            return True
            
        except Exception as e:
            app_logger.error(f"Cache delete error: {str(e)}")
            return False

    def clear_all(self) -> bool:
        """모든 캐시 데이터 삭제"""
        if not self.redis_client:
            return False

        try:
            self.redis_client.flushdb()
            app_logger.info("All cache data cleared")
            return True
            
        except Exception as e:
            app_logger.error(f"Cache clear error: {str(e)}")
            return False 