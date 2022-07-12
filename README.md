# NExLAB_prj
0. 실행 환경

    python 3.8.8
    
    CUDA 10.2
    
    pytorch 1.8.0
    
    torchvision 0.9.0
    
    package: numpy, pandas, tqdm, scikit-learn, scipy, matplotlib 등 
    

1. config 폴더 설명

    1.1 config.ini : configuration parameter 설정 파일

    1.2 case_info_160.csv : 환자 정보 파일
  
  
2. py 파일 설명

    2.1 analyzer.py : 시각화 및 통계, 각종 추가 분석

    2.2 common.py : config 파일 처리, file I/O, batch process

    2.3 dl_core.py : 각종 deep learning network 구현

    2.4 dl_manager.py : 네트워크 학습, 성능 평가

    2.5 main.py : 메인

    2.6 preprocess.py : 전처리 알고리즘

    2.7 spectrum_loader.py : custom file loader, data augmentation

    2.8 statistics.py : 각종 통계 관련 
    
    
 3. 실행 방법
 
     raw 데이터를 준비하고 main.py를 실행합니다
     
     메인함수에서 전처리, 모델 학습, 성능 평가 별로 각각의 함수를 호출하는 식입니다.
     
     전처리는 preprocess(), 모델 학습은 train_model(), 성능평가는 evaluate_model() 입니다.
     
     이해를 돕기 위해 각 함수 내에 parameter 별로 주석을 달아놨습니다.
     
     
