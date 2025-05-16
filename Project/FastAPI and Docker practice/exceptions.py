# 에러 처리: 상세한 에러 메시지와 적절한 HTTP 상태 코드 반환
# 1. RecommendationError (기본 예외)
#    - HTTP 500 에러 (서버 내부 오류)
#    - 모든 추천 시스템 관련 에러의 기본 클래스

# 2. DataLoadError (데이터 로드 예외)
#    - RecommendationError 상속
#    - 데이터 로딩 실패 시 발생
#    - 기본 메시지: "데이터를 로드하는 중 오류가 발생했습니다."

# 3. ModelInitializationError (모델 초기화 예외)
#    - RecommendationError 상속
#    - 모델 초기화 실패 시 발생
#    - 기본 메시지: "모델 초기화 중 오류가 발생했습니다."

# 4. InvalidInputError (입력 검증 예외)
#    - HTTP 400 에러 (잘못된 요청)
#    - 사용자 입력 검증 실패 시 발생
#    - 커스텀 메시지 필요

from fastapi import HTTPException, status

class RecommendationError(HTTPException):
    """추천 시스템 관련 기본 예외 클래스"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

class DataLoadError(RecommendationError):
    """데이터 로드 관련 예외"""
    def __init__(self, detail: str = "데이터를 로드하는 중 오류가 발생했습니다."):
        super().__init__(detail=detail)

class ModelInitializationError(RecommendationError):
    """모델 초기화 관련 예외"""
    def __init__(self, detail: str = "모델 초기화 중 오류가 발생했습니다."):
        super().__init__(detail=detail)

class InvalidInputError(HTTPException):
    """잘못된 입력 관련 예외"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        ) 