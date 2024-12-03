import os
import sys

# 현재 디렉토리의 상위 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))  # 상위 경로 추가

# dev.PSW.Line_find.py에서 Line_find 클래스를 가져오기
from dev.PSW.DL.Line_find import Line_find

# Line_find 클래스의 Print 메서드 호출
line_find = Line_find
line_find.Print()
