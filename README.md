# NExLAB_prj

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
