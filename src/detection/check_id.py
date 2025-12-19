import json

ANNOTATION_FILE = "./data/train_annotations.json"

def main():
    try:
        with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            categories = data['categories']
            
            print(f"--- {ANNOTATION_FILE} 내부 ID 확인 ---")
            print(f"총 카테고리 개수: {len(categories)}개")
            print("상위 5개 카테고리 정보:")
            
            for i, cat in enumerate(categories[:5]):
                print(f"순서 {i+1}: 이름(name)='{cat['name']}', ID='{cat['id']}'")
                
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()