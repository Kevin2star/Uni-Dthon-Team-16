"""공통 경로. 데이터 위치는 환경변수 DATA_DIR로 바꿀 수 있습니다.

    export DATA_DIR=/path/to/battery_data_light   # train/ validation/ test/ sample_submission.csv 가 있는 폴더
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get('DATA_DIR', '/home/ubuntu/data/battery_data_light'))
