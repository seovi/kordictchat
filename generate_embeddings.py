"""
CSV 사전 데이터에 대한 임베딩을 생성하는 스크립트
"""
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import os

def generate_embeddings(input_file, output_file, model_name='distiluse-base-multilingual-cased-v1', batch_size=32):
    """
    CSV 파일에서 단어와 정의에 대한 임베딩을 생성하고 결과를 저장합니다.
    
    Args:
        input_file (str): 입력 CSV 파일 경로
        output_file (str): 출력 CSV 파일 경로
        model_name (str): 사용할 SentenceTransformer 모델 이름
        batch_size (int): 한 번에 처리할 데이터 수
    """
    print(f"'{input_file}' 파일 로드 중...")
    
    try:
        # CSV 파일 로드
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"데이터 로드 완료: {len(df)} 항목")
        print(f"컬럼: {df.columns.tolist()}")
    except Exception as e:
        print(f"CSV 파일 로드 오류: {e}")
        # 특수 문자나 따옴표 등으로 인한 오류일 수 있으므로 다른 방법 시도
        try:
            df = pd.read_csv(input_file, encoding='utf-8', quoting=1)
            print(f"quoting=1 옵션으로 데이터 로드 완료: {len(df)} 항목")
        except Exception as e2:
            print(f"두 번째 로드 시도 오류: {e2}")
            try:
                df = pd.read_csv(input_file, encoding='utf-8', engine='python')
                print(f"engine='python' 옵션으로 데이터 로드 완료: {len(df)} 항목")
            except Exception as e3:
                print(f"모든 로드 시도 실패: {e3}")
                return
    
    # 임베딩 모델 로드
    print(f"모델 '{model_name}' 로드 중...")
    model = SentenceTransformer(model_name)
    print(f"모델 로드 완료. 벡터 차원: {model.get_sentence_embedding_dimension()}")
    
    # 임베딩을 생성할 텍스트 준비
    texts = []
    for _, row in df.iterrows():
        # 단어와 정의를 결합하여 임베딩 생성
        text = f"{row['word']}: {row['definition']}"
        texts.append(text)
    
    # 배치 처리로 임베딩 생성
    print("임베딩 생성 중...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        all_embeddings.extend(batch_embeddings.tolist())
    
    # 임베딩 JSON 문자열로 변환하여 데이터프레임에 추가
    df['embedding'] = [json.dumps(emb) for emb in all_embeddings]
    
    # 결과 저장
    print(f"결과를 '{output_file}'에 저장 중...")
    df.to_csv(output_file, index=False)
    
    # Numpy 배열로도 저장 (scikit-learn과 함께 사용하기 위함)
    embeddings_array = np.array(all_embeddings)
    np.save(os.path.splitext(output_file)[0] + '_embeddings.npy', embeddings_array)
    
    print(f"임베딩 생성 완료! 생성된 임베딩 차원: {len(all_embeddings[0])}")
    print(f"CSV 파일 저장 완료: {output_file}")
    print(f"NumPy 배열 저장 완료: {os.path.splitext(output_file)[0] + '_embeddings.npy'}")
    
    # 첫 번째 임베딩 샘플 확인
    print("\n첫 번째 임베딩 샘플:")
    sample_vector = all_embeddings[0][:5]  # 처음 5개 값만 표시
    print(f"{sample_vector} ... (총 {len(all_embeddings[0])} 차원)")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단어 사전 데이터에 대한 임베딩을 생성합니다.")
    parser.add_argument("--input", default="dictionary.csv", help="입력 CSV 파일 경로")
    parser.add_argument("--output", default="dictionary_with_embeddings.csv", help="출력 CSV 파일 경로")
    parser.add_argument("--model", default="distiluse-base-multilingual-cased-v1", 
                        help="사용할 SentenceTransformer 모델 이름")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    
    args = parser.parse_args()
    
    generate_embeddings(args.input, args.output, args.model, args.batch_size)