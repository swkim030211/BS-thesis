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

# 색상 스케일 (수정된 Charge_dispersion에 맞게 조정)
CHARGE_DISP_MIN = 1e-20  # 로그 스케일 최소값
CHARGE_DISP_MAX = 1000.0  # 최대값 (수정된 값 범위 고려)

# CSV 파일 디렉토리
csv_dir = r'C:\Users\User\PycharmProjects\QuTiP'
csv_pattern = os.path.join(csv_dir, "*fidelity_and_leakage.csv")

# 데이터 수집
data_points = []

# CSV 파일 찾기
csv_files = glob.glob(csv_pattern)
if not csv_files:
    logger.error(f"{csv_dir}에 CSV 파일이 없습니다.")
else:
    logger.info(f"{len(csv_files)}개의 CSV 파일을 찾았습니다.")
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        logger.debug(f"파일 처리 중: {filename}")

        # 파일 이름 파싱
        try:
            parts = filename.replace('fidelity_and_leakage.csv', '').split('_')
            if len(parts) >= 8:
                ec_val = float(parts[0].replace('Ec', ''))  # Ec 값 추출
                lj_start = float(parts[1].replace('Lj', ''))
                lj_end = float(parts[2])
                pad_gap_num = float(parts[3])
                pad_width = float(parts[4])  # y축
                pad_height = float(parts[5])  # x축
                pocket_width = float(parts[6])
                pocket_height = float(parts[7])
                logger.debug(f"파일 이름 파싱: Ec={ec_val}, Lj_start={lj_start}, Lj_end={lj_end}, "
                             f"pad_gap={pad_gap_num}, pad_width={pad_width}, pad_height={pad_height}, "
                             f"pocket_width={pocket_width}, pocket_height={pocket_height}")
            else:
                logger.warning(f"잘못된 파일 이름: {filename}")
                continue
        except ValueError as e:
            logger.error(f"파일 이름 파싱 오류 {filename}: {e}")
            continue

        # CSV 읽기
        try:
            df = pd.read_csv(file_path)
            logger.debug(f"CSV 읽기 완료 {filename}, 컬럼: {list(df.columns)}")
        except Exception as e:
            logger.error(f"{filename} 읽기 오류: {e}")
            continue

        # 필요한 컬럼 확인
        required_columns = ['EJ_GHz', 'f_Q_GHz', 'Charge_dispersion', 'Ej/Ec']
        if all(col in df.columns for col in required_columns):
            for idx, row in df.iterrows():
                try:
                    ej = float(row['EJ_GHz'])
                    f_q = float(row['f_Q_GHz'])
                    charge_disp = float(row['Charge_dispersion'])
                    ej_ec = float(row['Ej/Ec'])
                    if not any(np.isnan([ej, f_q, charge_disp, ej_ec])):
                        # Charge_dispersion 수정: 절댓값 * 1000
                        charge_disp = abs(charge_disp) * 1000
                        if charge_disp < CHARGE_DISP_MIN:
                            charge_disp = CHARGE_DISP_MIN  # 로그 스케일 최소값 보장
                        data_points.append({
                            'pad_width': pad_width,
                            'pad_height': pad_height,
                            'ec_val': ec_val,  # Ec 값 추가
                            'charge_disp': charge_disp,
                            'lj_nh': float(row['Lj_nH'])  # Lj_nH 추가 (1D 플롯용)
                        })
                        logger.debug(f"행 {idx} in {filename}: pad_width={pad_width}, "
                                     f"pad_height={pad_height}, Ec={ec_val}, Charge_dispersion={charge_disp}, Lj_nH={row['Lj_nH']}")
                    else:
                        logger.warning(f"잘못된 데이터 스킵 in {filename}, 행 {idx}: "
                                       f"EJ_GHz={row['EJ_GHz']}, f_Q_GHz={row['f_Q_GHz']}, "
                                       f"Charge_dispersion={row['Charge_dispersion']}, Ej/Ec={row['Ej/Ec']}")
                except (ValueError, TypeError) as e:
                    logger.error(f"잘못된 데이터 in {filename}, 행 {idx}: {e}")
                    continue
        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"{filename}에 필요한 컬럼 누락: {missing_cols}. 발견된 컬럼: {list(df.columns)}")

# 데이터 처리 및 플롯
if data_points:
    logger.info(f"{len(data_points)}개의 유효 데이터 포인트 수집")
    data = pd.DataFrame(data_points)

    # pad_height, pad_width로 그룹화, 최소 Charge_dispersion 선택
    grouped_data = data.loc[data.groupby(['pad_height', 'pad_width'])['charge_disp'].idxmin()]
    logger.debug(f"그룹화된 데이터 (최소 Charge_dispersion):\n{grouped_data.head().to_string()}")

    # 플롯용 데이터 추출
    x = grouped_data['pad_height'].values  # x축: pad_height
    y = grouped_data['pad_width'].values  # y축: pad_width
    c = grouped_data['charge_disp'].values  # 최소 Charge_dispersion
    ec_vals = grouped_data['ec_val'].values  # Ec 값

    # x_unique, y_unique 유효성 검사
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    logger.debug(f"x_unique: {x_unique}, y_unique: {y_unique}")
    if len(x_unique) < 2 or len(y_unique) < 2:
        logger.warning("pad_height 또는 pad_width의 고유 값이 2개 미만입니다. 2D 플롯 불가. 1D 플롯으로 대체.")
        # 1D 플롯 (Lj_nH 대 Charge_dispersion)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['lj_nh'], data['charge_disp'], 'b.-', label='Charge Dispersion')
        ax.set_xlabel('Lj_nH')
        ax.set_ylabel('Charge Dispersion (x1000)')
        ax.set_title('Charge Dispersion vs Lj_nH')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.show()
        logger.info("1D 플롯 표시 성공")
    else:
        # 2D scatter 플롯 생성 (로그 스케일 Charge_dispersion)
        fig, ax = plt.subplots(figsize=(10, 8))
        try:
            norm = LogNorm(vmin=CHARGE_DISP_MIN, vmax=CHARGE_DISP_MAX)
            scatter = ax.scatter(x, y, c=c, cmap='viridis', s=50, marker='o', norm=norm)
            logger.info("Scatter 플롯 (로그 스케일 Charge_dispersion) 생성 성공")
        except Exception as e:
            logger.error(f"Scatter 플롯 생성 오류: {e}")
            raise

        # Ec 값에 따른 등고선 (다양한 Ec 값이 존재할 경우)
            #        try:
            #           if len(np.unique(ec_vals)) > 1:
            #        x_grid, y_grid = np.meshgrid(x_unique, y_unique)
            #       ec_grid = griddata((x, y), ec_vals, (x_grid, y_grid), method='linear')
            #         contour = ax.contour(x_grid, y_grid, ec_grid, levels=10, colors='blue', linestyles='dashed', alpha=0.7)
            #         ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f', colors='blue')
        #         logger.info("Ec 블루 등고선 추가 성공")
            #    except Exception as e:
            #        logger.error(f"Ec 등고선 생성 오류: {e}")
            logger.info("Ec 등고선 없이 진행")
        #
        # 레이블, 제목, 그리드 설정
        ax.set_xlabel('Pad Height')
        ax.set_ylabel('Pad Width')
        ax.set_title('Charge Dispersion vs Pad Height and Width (Log Scale)')
        ax.grid(True, linestyle='--', alpha=0.7)

        # 색상바 추가
        try:
            cbar = fig.colorbar(scatter, label='Min Charge Dispersion', format='%.1e')
            cbar.set_label('Min Charge Dispersion', fontsize=12)
            logger.info(f"색상바 추가 성공, Charge_dispersion 범위: {CHARGE_DISP_MIN} to {CHARGE_DISP_MAX}")
        except Exception as e:
            logger.error(f"색상바 추가 오류: {e}")
            raise

        # 플롯 표시
        try:
            plt.show()
            logger.info("플롯 표시 성공")
        except Exception as e:
            logger.error(f"플롯 표시 오류: {e}")
            raise
else:
    logger.error("유효한 데이터 포인트가 없습니다. CSV 파일, 컬럼 이름, 데이터 유효성을 확인하세요.")