import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# CSV 파일 디렉토리
csv_dir = r'C:\Users\User\PycharmProjects\QuTiP'
csv_pattern = os.path.join(csv_dir, "*fidelity_and_leakage.csv")

# 데이터 수집
data_points = []
for file_path in glob.glob(csv_pattern):
    filename = os.path.basename(file_path)
    logger.debug(f"파일 처리 중: {filename}")

    try:
        parts = filename.replace('fidelity_and_leakage.csv', '').split('_')
        ec_val = float(parts[0].replace('Ec', ''))  # Ec 값 추출 (MHz)
        lj_start = float(parts[1].replace('Lj', ''))
        lj_end = float(parts[2])
        pad_gap_num = float(parts[3])
        pad_width = float(parts[4])
        pad_height = float(parts[5])
        data_points.append({
            'pad_width': pad_width,
            'pad_height': pad_height,
            'ec_val': ec_val
        })
    except Exception as e:
        logger.error(f"파일 처리 오류 {filename}: {e}")
        continue

data = pd.DataFrame(data_points)
logger.info(f"총 {len(data)}개의 데이터 포인트 수집")

# 데이터가 충분한지 확인
if len(data) == 0:
    logger.error("데이터 포인트가 없습니다. CSV 파일을 확인하세요.")
    exit()

# 보간을 위한 데이터 준비
x = data['pad_height'].values
y = data['pad_width'].values
z = data['ec_val'].values

# 로그 스케일 최소값 보정 (0에 가까운 값 방지)
z = np.where(z <= 0, np.nanmin(z[z > 0]) / 10, z)  # 0 이하 또는 0이면 최소 양수 값의 1/10으로 대체

# 그리드 생성
xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='linear', fill_value=np.nan)

# Heatmap 플롯 (로그 스케일)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(zi, extent=[min(x), max(x), min(y), max(y)], aspect='auto', cmap='viridis',
               norm=LogNorm(vmin=np.nanmin(zi[zi > 0]), vmax=np.nanmax(zi)))
ax.set_title('Ec (Log Scale) vs Pad Height and Pad Width')
ax.set_xlabel('Pad Height')
ax.set_ylabel('Pad Width')

# 색상바 추가 (로그 스케일)
plt.colorbar(im, label='Ec (MHz)')

# 데이터 포인트 표시 (선택 사항)
ax.scatter(x, y, c='red', s=10, alpha=0.5)

plt.tight_layout()
plt.show()