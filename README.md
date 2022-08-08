# 악성 댓글 분류 (Malicious Comment Classification)

#### 개발 인원: 3명

#### 담당 역할 
- 데이터 수집 및 라벨링
- 모델링

#### 기술 스택 
- Language: Python
- Modeling Framework :Pytorch
- Web Framework: Flask

## 프로젝트 상세 내용
- 문제 정의 및 주제 선정

비속어 뿐만 아니라 맥락까지 고려하여 악성 댓글을 분류해 낼 수 있는 시스템
- 데이터셋 구축
    - 데이터 출처: Korean HateSpeech Dataset, APEACH Datasets, 각 종 사이트 댓글 크롤링 후 선별
    - 비방/차별/성적 수치심 유발/저주◦협박으로 멀티 라벨링
    - 악성댓글의 세부적인 항목 분류를 통해 악성 댓글 제재 시 당위성을 부여하고자 함
    - 모델이 혼동하지 않는 명확한 악성댓글 분류 기준을 정하는 것이 어려운 부분
      → 타 연구에서 사용한 악성댓글 분류 기준을 참고하고 의논을 통해 분류 기준을 세움
      → 팀원 3명이서 교차 검증을 통해 라벨링을 진행하고 의견이 일치하지 않는 모호한 코퍼스  
          는 제외
    - 데이터 불균형 문제
    - 비방에 비해 차별/성적 수치심 유발/저주◦협박의 데이터 양은 많지 않음
    - 오버 샘플링으로 데이터 불균형 문제 완화
- 모델 설계
    - SKT에서 공개한 Pretrained 모델인 KoBERT를 선택
    - 세부적으로 라벨링 된 데이터가 있지만 데이터 양이 적은 편이여서 예측을 하기전에 모델이 언어의 이해도를 갖고 있다면 좋을 것이라 판단
    - 비속어가 존재하지 않더라도 문장의 맥락을 보고 악성댓글 여부를 판단하는 것이 목적
- 평가 지표
    - 설문조사를 통해 사용자들은 악성댓글을 사용하지 않았는데 악성댓글 작성자로 분류될 때
    더 민감하게 반응함을 파악  → Precision을 주요 평가 지표로 선정
    - 표현의 자유를 최대한 보장


## Structure
```
Malicious-Comment-Classification-System
├─ .gitignore
├─ data
├─ dataloader
├─ KoBERT
├─ model
├─ preprocessing
├─ README.md
├─ service
└─ train
```

## How to install KoBERT 
https://github.com/SKTBrain/KoBERT

```
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT
python setup.py install
```
