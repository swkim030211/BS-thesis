import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# 1. 계산 함수 정의 (변경 없음)
def charge_dispersion(ej, ec):
    """트랜스몬의 전하 분산 크기를 계산합니다."""
    ec = np.where(ec == 0, 1e-9, ec)
    ratio = ej / ec
    term1 = np.sqrt(2 / np.pi)
    term2 = (8 * ratio)**(0.25)
    term3 = np.exp(-np.sqrt(8 * ratio))
    return ec * term1 * term2 * term3

def qubit_frequency(ej, ec):
    """트랜스몬의 큐비트 주파수를 계산합니다."""
    return np.sqrt(8 * ej * ec) - ec

# 2. 그래프를 위한 E_J와 E_C 값의 범위 설정 (변경 없음)
ej_vals = np.linspace(1, 50, 400)
ec_vals = np.linspace(0.05, 1.0, 400)

# =================== 수정된 부분 1 ===================
# meshgrid의 생성 순서를 바꿔 x축이 EC, y축이 EJ가 되도록 합니다.
EC, EJ = np.meshgrid(ec_vals, ej_vals)
# =================================================

# 3. 각 그리드 포인트에서 값 계산 (변경 없음)
Z_dispersion = charge_dispersion(EJ, EC)
FQ = qubit_frequency(EJ, EC) # 큐비트 주파수 계산
RATIO = EJ / EC             # Ej/Ec 비율 계산

# 4. 등고선 그래프 생성
fig, ax = plt.subplots(figsize=(12, 9))

# =================== 수정된 부분 2 ===================
# 배경: 전하 분산 등고선 (채우기) - 인수 순서 변경
levels = np.logspace(np.log10(Z_dispersion.min()), np.log10(Z_dispersion.max()), 20)
contour = ax.contourf(EC, EJ, Z_dispersion, levels=levels, cmap='plasma', norm=LogNorm())
cbar = fig.colorbar(contour)
cbar.set_label('Charge Dispersion |δE₀| (GHz, log scale)')

# 등고선 1: E_J/E_C 비율 (빨간색 실선) - 인수 순서 변경
ratio_levels = [10, 20, 30, 40, 50, 60, 70, 80, 150]
ratio_contour = ax.contour(EC, EJ, RATIO, levels=ratio_levels, colors='red', linewidths=1.5)
ax.clabel(ratio_contour, inline=True, fontsize=10, fmt='Ej/Ec = %d')

# 등고선 2: 큐비트 주파수 (파란색 점선) - 인수 순서 변경
fq_levels = np.arange(3.0, 9.0, 1.0) # 3GHz부터 8GHz까지 1GHz 간격
fq_contour = ax.contour(EC, EJ, FQ, levels=fq_levels, colors='blue', linestyles='--')
ax.clabel(fq_contour, inline=True, fontsize=10, fmt='%.1f GHz')
# =================================================

# 5. 축과 제목 레이블 설정
# =================== 수정된 부분 3 ===================
ax.set_xlabel('$E_C$ (GHz)', fontsize=14) # x축 레이블
ax.set_ylabel('$E_J$ (GHz)', fontsize=14) # y축 레이블
# =================================================
ax.set_title('Charge dispersion per $E_J$ and $E_C$', fontsize=16, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)

# =================== 수정된 부분 4 ===================
ax.set_xlim(ec_vals.min(), ec_vals.max()) # x축 범위
ax.set_ylim(ej_vals.min(), ej_vals.max()) # y축 범위
# =================================================

# 그래프 출력
plt.show()