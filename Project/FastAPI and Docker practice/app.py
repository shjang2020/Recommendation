from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Optional
from data_loader import MovieLensDataLoader
from recommender import BertBasedRecommender
from logger import app_logger
from exceptions import RecommendationError, DataLoadError, ModelInitializationError, InvalidInputError
from cache import CacheManager
import warnings
import logging
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get TMDB API key from environment variable
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# 경고 메시지 필터링
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Some weights of the model checkpoint')

# transformers 라이브러리의 로그 레벨 설정
logging.getLogger("transformers").setLevel(logging.ERROR)

app = FastAPI(
    title="Movie Recommendation API",
    description="""영화 추천 시스템 API
    \n 영화 장르 종류는 아래와 같습니다.
    \n Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Fantasy, Horror, Mystery, Romance, Science Fiction, Thriller, War, Western""",
    version="1.0.1"
)

# 전역 변수로 recommender와 cache_manager 인스턴스 저장
recommender = None
cache_manager = None

class RecommendationRequest(BaseModel):
    genres: Optional[List[str]] = None
    keywords: Optional[str] = None
    num_recommendations: Optional[int] = 1
    start_year: Optional[int] = 1900
    end_year: Optional[int] = 2020

    @validator('num_recommendations')
    def validate_num_recommendations(cls, v):
        if v < 1 or v > 10:
            raise InvalidInputError("추천 영화 수는 1에서 10 사이여야 합니다.")
        return v

    @validator('genres')
    def validate_list_length(cls, v):
        if v is not None and len(v) > 10:
            raise InvalidInputError("장르는는 최대 10개까지 입력 가능합니다.")
        return v

    @validator('keywords')
    def validate_list_length(cls, v):
        if v is not None and len(v) > 100:
            raise InvalidInputError("키워드는 최대 100자까지 입력 가능합니다.")
        return v
    
    @validator('start_year', 'end_year')
    def validate_year(cls, v):
        if v is not None and v < 1900:
            raise InvalidInputError("연도는 1900년 이후여야 합니다.")
        return v    
    
class MovieRecommendation(BaseModel):
    recommendation_reason: str
    movie_title: str
    genres: List[str]
    similarity_score: float
    plot: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    app_logger.info(
        f"Method: {request.method} Path: {request.url.path} "
        f"Status: {response.status_code} Duration: {process_time:.2f}s"
    )
    
    return response

@app.exception_handler(RecommendationError)
async def recommendation_exception_handler(request: Request, exc: RecommendationError):
    """추천 시스템 예외 처리"""
    app_logger.error(f"Recommendation error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(InvalidInputError)
async def invalid_input_exception_handler(request: Request, exc: InvalidInputError):
    """잘못된 입력 예외 처리"""
    app_logger.warning(f"Invalid input: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 추천 시스템 초기화"""
    global recommender, cache_manager
    try:
        app_logger.info("Starting recommendation system initialization...")
        
        # 캐시 매니저 초기화 (기존 캐시 유지)
        # 캐시 수동 초기화 : bash - curl -X DELETE "http://localhost:8000/cache"
        #                  Redis CLI 사용 - redis-cli
        cache_manager = CacheManager()
        if cache_manager.redis_client:
            app_logger.info("Redis 캐시 연결 성공 - 기존 캐시 유지")
        else:
            app_logger.warning("Redis 캐시 연결 실패 - 캐시 기능 비활성화")
        
        # 데이터 로더 초기화
        loader = MovieLensDataLoader(tmdb_api_key=TMDB_API_KEY)
        
        # 데이터 로드
        movies_df, ratings_df = loader.load_processed_data()
        if movies_df is None:
            app_logger.info("No processed data found. Downloading new data...")
            loader.download_data()
            movies_df, ratings_df = loader.load_data()
            
            # 데이터 검증
            app_logger.info("Validating raw data...")
            if not loader.validate_data(movies_df, ratings_df):
                app_logger.warning("Data validation found issues, but proceeding with processing...")
            
            # 데이터 전처리
            movies_df = loader.prepare_movie_features(movies_df, ratings_df)
            
            # 처리된 데이터 검증
            app_logger.info("Validating processed data...")
            if not loader.validate_data(movies_df, ratings_df):
                app_logger.warning("Processed data validation found issues, but proceeding...")
            
            loader.save_processed_data(movies_df, ratings_df)
        
        # 임베딩 로드
        genre_embeddings, plot_embeddings = loader.load_embeddings()
        
        # 추천 시스템 초기화
        app_logger.info("Initializing recommender system...")
        recommender = BertBasedRecommender()
        
        if genre_embeddings is None or plot_embeddings is None:
            app_logger.info("No saved genre or plot embeddings found. Generating new embeddings...")
            recommender.fit(movies_df, genre_embeddings, plot_embeddings)
            loader.save_embeddings(recommender.genre_embeddings, recommender.plot_embeddings)
        else:
            recommender.fit(movies_df, genre_embeddings, plot_embeddings)
            
        app_logger.info("Recommendation system initialization completed successfully!")
        print("Press CTRL+C to quit --> http://localhost:8000/docs")
        
    except Exception as e:
        app_logger.error(f"Error during system initialization: {str(e)}")
        raise ModelInitializationError(f"시스템 초기화 중 오류가 발생했습니다: {str(e)}")

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    app_logger.info("Root endpoint accessed")
    return {
        "message": "영화 추천 시스템 API에 오신 것을 환영합니다!",
        "endpoints": {
            "/recommend": "영화 추천을 받을 수 있는 엔드포인트",
            "/docs": "API 문서"
        }
    }

@app.post("/recommend", response_model=List[MovieRecommendation])
async def get_recommendations(request: RecommendationRequest):
    """영화 추천 엔드포인트"""
    try:
        app_logger.info(f"Recommendation request received - Genres: {request.genres}, Keywords: {request.keywords}")
        
        # 캐시에서 데이터 조회
        if cache_manager:
            cached_result = cache_manager.get(
                request.genres,
                request.keywords,
                request.num_recommendations,
                request.start_year,
                request.end_year
            )
            if cached_result:
                app_logger.info("Returning cached recommendations")
                return cached_result
        
        # 캐시에 없으면 새로운 추천 생성
        app_logger.info(f"Generating recommendations - Genres: {request.genres}, Keywords: {request.keywords}")
        recommendations = recommender.recommend(
            request.genres,
            request.keywords,
            request.num_recommendations,
            request.start_year,
            request.end_year
        )
        
        # 디버깅을 위한 응답 데이터 로깅
        app_logger.debug(f"Generated recommendations: {recommendations.to_dict('records')}")
        
        # 추천 이유 설정
        if not request.genres and not request.keywords:
            recommendations['recommendation_reason'] = "선호하는 장르나 키워드가 없어서 랜덤으로 추천해드립니다."
        elif request.genres and not request.keywords:
            recommendations['recommendation_reason'] = f"{request.start_year}년 ~ {request.end_year}년 사이의 영화 중 선호하신 장르 '{', '.join(request.genres)}'를 고려한 영화를 추천해드립니다."
        elif not request.genres and request.keywords:
            recommendations['recommendation_reason'] = f"{request.start_year}년 ~ {request.end_year}년 사이의 영화 중 입력하신 키워드 '{request.keywords}'를 고려한 영화를 추천해드립니다."
        else:
            recommendations['recommendation_reason'] = f"{request.start_year}년 ~ {request.end_year}년 사이의 영화 중 선호하신 장르 '{', '.join(request.genres)}'와 키워드 '{request.keywords}'를 모두 고려한 영화를 추천해드립니다."
        
        # 포스터 URL 생성
        recommendations['poster_url'] = recommendations['poster_path'].apply(
            lambda x: f"https://image.tmdb.org/t/p/w500{x}" if x else None
        )
        recommendations['backdrop_url'] = recommendations['backdrop_path'].apply(
            lambda x: f"https://image.tmdb.org/t/p/original{x}" if x else None
        )
        recommendations['movie_title'] = recommendations['clean_title'] + ' (' + recommendations['year'].astype(str) + ')'
        # genres를 리스트로 유지
        recommendations['genres'] = recommendations['genres'].apply(
            lambda x: x if isinstance(x, list) else [g.strip() for g in x.split(',')]
        )
        recommendations = recommendations[['recommendation_reason', 'movie_title', 'genres', 'similarity_score', 'plot',
                                           'poster_url', 'backdrop_url']]
        # 결과를 캐시에 저장
        if cache_manager:
            cache_manager.set(
                request.genres,
                request.keywords,
                request.num_recommendations,
                request.start_year,
                request.end_year,
                recommendations.to_dict('records')
            )
        
        app_logger.info(f"Successfully generated {len(recommendations)} recommendations")
        return recommendations.to_dict('records')
        
    except Exception as e:
        app_logger.error(f"Error generating recommendations: {str(e)}")
        raise RecommendationError(f"추천 생성 중 오류가 발생했습니다: {str(e)}")

@app.delete("/cache")
async def clear_cache():
    """캐시 초기화 엔드포인트"""
    if not cache_manager:
        raise RecommendationError("캐시 시스템이 초기화되지 않았습니다.")
    
    if cache_manager.clear_all():
        return {"message": "캐시가 성공적으로 초기화되었습니다."}
    else:
        raise RecommendationError("캐시 초기화 중 오류가 발생했습니다.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
