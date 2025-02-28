"""
scikit-learn을 활용한 벡터 검색 구현
"""
import numpy as np
import pandas as pd
import json
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class VectorSearch:
    def __init__(self, model_name='distiluse-base-multilingual-cased-v1'):
        """
        벡터 검색 엔진 초기화
        
        Args:
            model_name (str): 사용할 SentenceTransformer 모델 이름
        """
        self.df = None
        self.embeddings = None
        self.model = None
        self.model_name = model_name
        print(f"벡터 검색 엔진 초기화 (모델: {model_name})")
    
    def load_model(self):
        """
        텍스트 임베딩 모델 로드
        """
        if self.model is None:
            print(f"모델 '{self.model_name}' 로드 중...")
            self.model = SentenceTransformer(self.model_name)
            print(f"모델 로드 완료. 벡터 차원: {self.model.get_sentence_embedding_dimension()}")
        return self.model
    
    def load_data(self, csv_file, embeddings_file=None):
        """
        CSV 파일과 임베딩 파일 로드
        
        Args:
            csv_file (str): 데이터가 포함된 CSV 파일 경로
            embeddings_file (str, optional): 임베딩이 저장된 .npy 파일 경로
        """
        print(f"'{csv_file}' 파일 로드 중...")
        
        try:
            # CSV 파일 로드
            self.df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"데이터 로드 완료: {len(self.df)} 항목")
            
            # 임베딩 로드 또는 추출
            if embeddings_file and os.path.exists(embeddings_file):
                # 별도의 .npy 파일에서 임베딩 로드
                self.embeddings = np.load(embeddings_file)
                print(f"임베딩 로드 완료: {self.embeddings.shape}")
            elif 'embedding' in self.df.columns:
                # CSV에서 임베딩 컬럼 추출
                print("CSV 파일에서 임베딩 추출 중...")
                embeddings_list = []
                for emb_str in tqdm(self.df['embedding']):
                    embeddings_list.append(np.array(json.loads(emb_str)))
                self.embeddings = np.vstack(embeddings_list)
                print(f"임베딩 추출 완료: {self.embeddings.shape}")
            else:
                print("경고: 임베딩 데이터를 찾을 수 없습니다. 데이터를 검색하려면 임베딩을 생성해야 합니다.")
                return False
            
            return True
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return False
    
    def generate_query_embedding(self, query):
        """
        쿼리 텍스트에 대한 임베딩 생성
        
        Args:
            query (str): 검색 쿼리 텍스트
        
        Returns:
            numpy.ndarray: 쿼리 임베딩 벡터
        """
        if self.model is None:
            self.load_model()
        
        query_embedding = self.model.encode([query])[0]
        return query_embedding
    
    def search(self, query, top_k=5):
        """
        쿼리와 가장 유사한 항목 검색
        
        Args:
            query (str): 검색 쿼리 텍스트
            top_k (int): 반환할 결과 수
        
        Returns:
            list: 검색 결과 리스트 (각 항목은 딕셔너리)
        """
        if self.embeddings is None or self.df is None:
            print("오류: 먼저 데이터를 로드해야 합니다.")
            return []
        
        # 쿼리 임베딩 생성
        query_embedding = self.generate_query_embedding(query)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 상위 k개 항목 가져오기
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # 결과 구성
        results = []
        for i, idx in enumerate(top_indices):
            item = self.df.iloc[idx].to_dict()
            
            # 임베딩 필드가 있으면 제외 (너무 크기 때문)
            if 'embedding' in item:
                del item['embedding']
            
            # 유사도 점수 추가
            item['relevance'] = float(similarities[idx])
            results.append(item)
        
        return results
    
    def save(self, file_path="vector_search_data"):
        """
        벡터 검색 데이터 저장
        
        Args:
            file_path (str): 저장할 디렉토리 경로
        """
        os.makedirs(file_path, exist_ok=True)
        
        # 데이터프레임 저장 (embedding 컬럼 제외)
        if self.df is not None:
            if 'embedding' in self.df.columns:
                df_save = self.df.drop(columns=['embedding'])
            else:
                df_save = self.df
            df_save.to_pickle(os.path.join(file_path, "data.pkl"))
        
        # 임베딩 배열 저장
        if self.embeddings is not None:
            np.save(os.path.join(file_path, "embeddings.npy"), self.embeddings)
        
        # 모델 정보 저장
        with open(os.path.join(file_path, "model_info.pkl"), "wb") as f:
            pickle.dump({"model_name": self.model_name}, f)
        
        print(f"벡터 검색 데이터가 '{file_path}' 디렉토리에 저장되었습니다.")
    
    def load(self, file_path="vector_search_data"):
        """
        저장된 벡터 검색 데이터 로드
        
        Args:
            file_path (str): 저장된 데이터가 있는 디렉토리 경로
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 데이터프레임 로드
            self.df = pd.read_pickle(os.path.join(file_path, "data.pkl"))
            
            # 임베딩 배열 로드
            self.embeddings = np.load(os.path.join(file_path, "embeddings.npy"))
            
            # 모델 정보 로드
            with open(os.path.join(file_path, "model_info.pkl"), "rb") as f:
                model_info = pickle.load(f)
                self.model_name = model_info["model_name"]
            
            print(f"벡터 검색 데이터를 '{file_path}'에서 로드했습니다.")
            print(f"데이터: {len(self.df)}개 항목, 임베딩: {self.embeddings.shape}")
            return True
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return False

# 테스트 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="벡터 검색 엔진 테스트")
    parser.add_argument("--load-only", action="store_true",
                        help="저장된 데이터만 로드 (CSV 파일 처리 건너뛰기)")
    parser.add_argument("--csv", default="dictionary_with_embeddings.csv",
                        help="임베딩이 포함된 CSV 파일 경로")
    parser.add_argument("--embeddings", default=None,
                        help="임베딩 .npy 파일 경로 (기본값: CSV 파일 이름에서 유추)")
    parser.add_argument("--query", default="인공지능",
                        help="테스트 검색 쿼리")
    parser.add_argument("--top-k", type=int, default=5,
                        help="반환할 검색 결과 수")
    
    args = parser.parse_args()
    
    # 벡터 검색 객체 생성
    search_engine = VectorSearch()
    
    if args.load_only:
        # 저장된 데이터 로드
        success = search_engine.load("vector_search_data")
        if not success:
            print("저장된 데이터를 찾을 수 없습니다. CSV 파일에서 로드합니다.")
            args.load_only = False
    
    if not args.load_only:
        # 모델 로드
        search_engine.load_model()
        
        # 임베딩 파일 경로가 지정되지 않은 경우 CSV 파일 이름에서 유추
        if args.embeddings is None:
            args.embeddings = os.path.splitext(args.csv)[0] + "_embeddings.npy"
            if not os.path.exists(args.embeddings):
                args.embeddings = None
        
        # 데이터 로드
        success = search_engine.load_data(args.csv, args.embeddings)
        if success:
            # 데이터 저장
            search_engine.save("vector_search_data")
    
    # 테스트 검색 수행
    print(f"\n'{args.query}'에 대한 검색 결과:")
    results = search_engine.search(args.query, args.top_k)
    
    for i, result in enumerate(results):
        print(f"[{i+1}] {result['word']} ({result['pos']}) - 관련도: {result['relevance']:.4f}")
        print(f"    분류: {result['cats']}")
        print(f"    정의: {result['definition']}")
        print()