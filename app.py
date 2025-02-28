"""
FastAPI 기반 사전 챗봇 API 서버
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List, Optional
import openai
from dotenv import load_dotenv
from sklearn_vector_search import VectorSearch

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="사전 챗봇 API", description="scikit-learn을 활용한 벡터 검색 기반 사전 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 구체적인 origin으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_rag: bool = True

class DictionaryEntry(BaseModel):
    target_code: int
    word: str
    cats: str
    pos: str
    definition: str
    relevance: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[DictionaryEntry]

# 벡터 검색 엔진
vector_search = None

# RAG를 통한 응답 생성 함수
async def generate_rag_response(query: str, results: List[dict]):
    system_prompt = """
    당신은 한국어 사전 챗봇입니다. 사용자의 질문에 대해 제공된 사전 정보를 바탕으로 답변하세요.
    제공된 정보가 없는 내용에 대해서는 '해당 정보를 찾을 수 없습니다'라고 답변하세요.
    답변은 간결하고 명확하게 제공하되, 정확한 사전적 정의와 함께 예시나 관련 정보를 포함하면 좋습니다.
    """
    
    context = "검색된 사전 항목:\n\n"
    for i, entry in enumerate(results):
        context += f"{i+1}. {entry['word']} [{entry['pos']}]: {entry['definition']}\n"
    
    try:
        if openai.api_key:
            # OpenAI API를 사용한 응답 생성
            response = await openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"질문: {query}\n\n{context}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        else:
            # OpenAI API가 설정되지 않은 경우 단순 응답
            best_match = results[0] if results else None
            if best_match:
                answer = f"'{query}'에 가장 관련된 단어는 '{best_match['word']}'입니다.\n\n"
                answer += f"정의: {best_match['definition']}\n"
                answer += f"품사: {best_match['pos']}\n"
                answer += f"분류: {best_match['cats']}"
                return answer
            else:
                return f"'{query}'에 대한 정보를 찾을 수 없습니다."
    
    except Exception as e:
        print(f"응답 생성 오류: {e}")
        # 오류 발생 시 기본 응답 제공
        if results:
            return f"'{query}'에 대한 검색 결과입니다. (OpenAI API 오류로 인해 자동 생성된 응답):\n\n{context}"
        else:
            return f"'{query}'에 대한 정보를 찾을 수 없습니다. 다른 검색어를 시도해보세요."

# 애플리케이션 시작 시 벡터 검색 엔진 초기화
@app.on_event("startup")
async def startup_event():
    global vector_search
    
    vector_search = VectorSearch()
    
    # 저장된 데이터가 있는지 확인
    if os.path.exists("vector_search_data"):
        success = vector_search.load("vector_search_data")
        if success:
            print("벡터 검색 데이터 로드 완료")
            return
    
    # 저장된 데이터가 없거나 로드 실패 시 새로 처리
    print("저장된 벡터 검색 데이터가 없습니다. 데이터 처리를 시작합니다.")
    
    # 임베딩 파일 경로 확인
    csv_file = "/Data/dictionary_with_embeddings.csv"
    embeddings_file = "/Data/dictionary_with_embeddings_embeddings.npy"
    
    if not os.path.exists(csv_file):
        # 임베딩이 포함된 CSV 파일이 없는 경우, 원본 CSV에서 임베딩 생성 필요
        if os.path.exists("dictionary.csv"):
            print("임베딩을 포함한 CSV 파일이 없습니다. 먼저 generate_embeddings.py를 실행하세요.")
        else:
            raise Exception("dictionary.csv 파일을 찾을 수 없습니다. 데이터 파일을 준비하세요.")
    
    # 데이터 로드
    vector_search.load_model()
    success = vector_search.load_data(csv_file, embeddings_file)
    
    if not success:
        raise Exception("벡터 검색 엔진 초기화에 실패했습니다.")
    
    # 데이터 저장
    vector_search.save("vector_search_data")
    print("벡터 검색 데이터 처리 및 저장 완료")

# 엔드포인트: 단어 검색
@app.post("/api/query", response_model=QueryResponse)
async def query_dictionary(request: QueryRequest):
    if vector_search is None:
        raise HTTPException(status_code=500, detail="벡터 검색 엔진이 초기화되지 않았습니다.")
    
    try:
        # 벡터 검색 수행
        results = vector_search.search(request.query, request.top_k)
        
        if not results:
            raise HTTPException(status_code=404, detail=f"'{request.query}'에 대한 결과를 찾을 수 없습니다.")
        
        # RAG를 통한 응답 생성 (선택적)
        if request.use_rag:
            answer = await generate_rag_response(request.query, results)
        else:
            # RAG를 사용하지 않는 경우 단순 응답
            answer = f"'{request.query}'에 대한 검색 결과입니다."
        
        # 응답 모델에 맞게 변환
        sources = [
            DictionaryEntry(
                target_code=int(entry['target_code']),
                word=entry['word'],
                cats=entry['cats'],
                pos=entry['pos'],
                definition=entry['definition'],
                relevance=float(entry['relevance'])
            )
            for entry in results
        ]
        
        return QueryResponse(answer=answer, sources=sources)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"검색 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 엔드포인트: 상태 확인
@app.get("/api/health")
async def health_check():
    status = "ok" if vector_search and vector_search.df is not None else "error"
    return {
        "status": status,
        "dictionary_entries": len(vector_search.df) if vector_search and vector_search.df is not None else 0,
        "embedding_dimension": vector_search.embeddings.shape[1] if vector_search and vector_search.embeddings is not None else 0
    }

# 메인 실행 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)