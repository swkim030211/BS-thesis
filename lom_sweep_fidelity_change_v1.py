import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal import MetalGUI, Dict, Headings
import pyEPR as epr
import numpy as np
import pandas as pd
import sys
from io import StringIO
import re
import qutip as qt
import matplotlib.pyplot as plt
from qutip import Options
from scipy.integrate import quad
from scipy.special import factorial
from qiskit_metal.analyses.quantization import LOManalysis
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
import pyaedt

h = 6.62607015e-34
Phi0 = 2.067833848e-15

def parse_lom_output(text: str) -> dict:
    patterns = {
        'f_Q_GHz': r"f_Q\s+([\d.-]+)\s+\[GHz\]",
        'EC_MHz': r"EC\s+([\d.-]+)\s+\[MHz\]",
        'EJ_GHz': r"EJ\s+([\d.-]+)\s+\[GHz\]",
        'alpha_MHz': r"alpha\s+([\d.-]+)\s+\[MHz\]",
        'Lq_nH': r"Lq\s+([\d.-]+)\s+\[nH\]",
        'Cq_fF': r"Cq\s+([\d.-]+)\s+\[fF\]",
        'T1_us': r"T1\s+([\d.-]+)\s+\[us\]",
        'gbus1_in_MHz': r"gbus1_in_MHz\s+([\d\.-]+)\s+\[MHz\]",
        'chi_bus1_MHz': r"χ_bus1\s+([\d\.-]+)\s+\[MHz\]"
    }

    data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            data[key] = float(match.group(1))
        else:
            data[key] = np.nan

    return data

#1. Design qiskit-metal
design = designs.DesignPlanar({}, True)
design.chips.main.size['size_x'] = '4mm'
design.chips.main.size['size_y'] = '4mm'
design.chips.main.material = 'silicon_loss'  # To change material of substrate, silicon is default value. 'Silicon_loss' is same as silicon but changed dielectric loss tangent as 1e-6
gui = MetalGUI(design)
design.delete_all_components()

q1 = TransmonPocket(design, 'Q1', options=dict(
    pad_gap='30 um', #30 um default, for siddiqi group 65
    pad_width='50 um',#455 um default, for siddiqi group 545 // SWK sweep 25 50 100 150 200 250 300 350 400 455 500 550 600
    pad_height = '175 um',#90 um default, for siddiqi group 135 // SWK sweep 10 25 50 90 100 125 150 175 200 225 250 Sweep of 90 175 225 are needed for Ej sweep
    pocket_width = '650 um',#650 um default, for siddiqi group 745
    pocket_height='650 um',#650 um default, for siddiqi group 535
    connection_pads=dict(
        readout=dict(loc_W=+1, loc_H=+1, pad_width='200um')
    )))
gui.rebuild()
gui.autoscale()

pad_gap_val = q1.options.pad_gap
pad_width_val = q1.options.pad_width
pad_height_val = q1.options.pad_height
pocket_width_val = q1.options.pocket_width
pocket_height_val = q1.options.pocket_height

pad_gap_num = float(pad_gap_val.split(' ')[0])
pad_width_num = float(pad_width_val.split(' ')[0])
pad_height_num = float(pad_height_val.split(' ')[0])
pocket_width_num = float(pocket_width_val.split(' ')[0])
pocket_height_num = float(pocket_height_val.split(' ')[0])




# 2. Q3D로 커패시턴스 매트릭스 계산 (1회 실행)
# =================================================================
c1 = LOManalysis(design, "q3d")
q3d = c1.sim.renderer
q3d.start()

q3d.activate_ansys_design("TransmonQubit_q3d", 'capacitive')
q3d.render_design(['Q1'], [('Q1', 'readout')])
q3d.analyze_setup("Setup")

c1.sim.capacitance_matrix, c1.sim.units = q3d.get_capacitance_matrix()
c1.sim.capacitance_all_passes, _ = q3d.get_capacitance_all_passes()
print("Capacitance Matrix (fF):")
print(c1.sim.capacitance_matrix)


# 3. LOM 분석 반복 실행 및 결과 파싱
# =================================================================
#lj_values = np.linspace(1, 500, 500)
Ej_values = np.linspace (5, 50, 1000)
k = (Phi0 / (2 * np.pi))**2 / h
lj_values = k/Ej_values
print(f"\n[INFO] Start Lj sweep: {lj_values} nH")
all_results_data = []

for lj in lj_values:
    print(f"\n---------- Lj = {lj:.2f} nH 분석 중... ----------")
    c1.setup.junctions = Dict(Lj=lj, Cj=2)
    c1.setup.freq_readout = 6.563 # 6.563 sweep
    c1.setup.freq_bus = []

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    c1.run_lom()

    sys.stdout = old_stdout
    lom_output_text = captured_output.getvalue()

    print(lom_output_text)

    parsed_data = parse_lom_output(lom_output_text)
    parsed_data['Lj_nH'] = lj
    all_results_data.append(parsed_data)

# 4. 최종 결과 취합 및 테이블 출력
# =================================================================
if all_results_data:
    final_summary_df = pd.DataFrame(all_results_data)

    # 열 순서 정리
    cols_order = ['Lj_nH', 'f_Q_GHz', 'alpha_MHz', 'EC_MHz', 'EJ_GHz', 'Lq_nH', 'Cq_fF', 'T1_us','gbus1_in_MHz', 'chi_bus1_MHz']
    final_summary_df = final_summary_df[cols_order]

    print("\n\n===================================================")
    print("      LOM Analysis Final Summary Table       ")
    print("===================================================")
    print(final_summary_df.to_string(index=False))
else:
    print("\n[ERROR] Error on parcing results.")

# 5. Ansys 종료
# =================================================================
c1.sim.close()

def calculate_tphi_ej_noise_vanharlingen(ej_ghz, ec_ghz, SI0_pA=50.0, t_exp_s=1.0):
    # 이 함수는 Ithier et al. (2005) Phys. Rev. B 72, 134519 논문 (특히 섹션 II C 1 및 III B 1)의 프레임워크와 경험적 값을 기반으로 구현되었습니다.
    # 트랜스몬 큐비트의 EJ 노이즈로 인한 위상 완화 시간(T_phi)을 계산합니다.
    # SI0_pA 및 t_exp_s 파라미터는 원래 함수 시그니처에 포함되어 있지만, 이 Ithier 기반 계산에서는 사용되지 않습니다.

    # 상수 정의
    h = 6.62607015e-34  # 플랑크 상수 (Joule*second)
    hbar = h / (2 * np.pi)  # 환산 플랑크 상수 (Joule*second)

    # ej_ghz 및 ec_ghz를 GHz 단위 에너지에서 Joule 단위 에너지로 변환
    # ej_ghz와 ec_ghz는 이미 GHz 단위의 에너지 값이므로, Joule로 변환하려면 h (J*s)와 1e9 (Hz/GHz)를 곱합니다.
    ej_joule = ej_ghz * 1e9 * h
    ec_joule = ec_ghz * 1e9 * h
    A_I = 0.2e-6
    # Ithier et al. (2005) 섹션 II C 1: 상대적 EJ 변동에 대한 경험적 스펙트럼 밀도 진폭, A_I = 0.2e-6
    # "Sc_dEj/Ej(|w|<2pi*10kHz) ~ (0.5e-6)^2 / |w|"
    # 이는 S_lambda(omega) = A / |omega| 형태에서 진폭 'A'가 (0.5e-6)^2임을 의미합니다.
    A_relative_EJ_noise = 2 * np.pi * (A_I)**2

    # 상대적 EJ 변동(lambda = delta EJ / EJ)에 대한 민감도 파라미터 D_lambda,z (rad/s) 계산
    # Ithier et al. 섹션 III B 1 및 표준 트랜스몬 해밀토니안에서 유도:
    # D_lambda,z = d(omega_01) / d(lambda) = EJ * d(omega_01) / d(EJ)
    # 트랜스몬의 큐비트 주파수 omega_01 근사: omega_01 = (1/hbar) * (sqrt(8 * EJ * EC) - EC)
    # d(omega_01)/d(EJ) = (1/hbar) * sqrt(2 * EC / EJ)
    # 따라서, D_lambda,z = EJ * (1/hbar) * sqrt(2 * EC / EJ) = (1/hbar) * sqrt(2 * EJ * EC)
    # 이 D_lambda,z의 단위는 rad/s입니다.

    # EJ 및 EC 값이 양수인지 확인하여 제곱근 오류 방지
    if ej_joule <= 0 or ec_joule <= 0:
        return np.inf

    D_lambda_z = (1 / hbar) * np.sqrt(2 * ej_joule * ec_joule)

    # Ithier et al. Eq. (22)에서 에코 감쇠(echo decay)를 위한 위상 완화 시간 T_phi_echo 계산:
    # f_z,E(t) = exp(-t^2 * D_lambda,z^2 * A * ln 2)
    # 이는 1/T_phi^2 = D_lambda,z^2 * A * ln 2를 의미합니다.
    # 따라서, T_phi = 1 / (abs(D_lambda,z) * sqrt(A * ln 2))

    if D_lambda_z == 0 or A_relative_EJ_noise == 0:
        t_phi_s = np.inf
    else:
        t_phi_s = 1 / (abs(D_lambda_z) * np.sqrt(A_relative_EJ_noise * np.log(2)))

    # 계산된 T_phi 값을 초(s)에서 나노초(ns)로 변환
    t_phi_ns = t_phi_s * 1e9

    return t_phi_ns


def calculate_tphi_from_charge_dispersion(ej_ghz, ec_ghz, A=1e-4):
    # https://doi.org/10.1103/PhysRevA.76.042319 eq 2.5
    """
    Koch et al. 2007 PRA 기반으로 transmon의 charge noise에 의한 dephasing time을 계산.
    """
    if ec_ghz == 0:
        return np.inf8

    m = 1
    # np.math.factorial is deprecated, use scipy.special.factorial
    prefactor = ((-1) ** m) * ec_ghz * (2 ** (4 * m + 5)) / factorial(m)

    ratio = ej_ghz / (2 * ec_ghz)
    if ratio < 0:  # Avoid complex numbers from negative roots
        return np.inf

    power = ratio ** (m / 2 + 3 / 4)
    exp_term = np.exp(-np.sqrt(8 * ej_ghz / ec_ghz))
    epsilon_1 = prefactor * np.sqrt(2 / np.pi) * power * exp_term

    gamma_phi_ghz = A * np.pi * abs(epsilon_1)

    # Convert from GHz to 1/ns
    gamma_phi_per_ns = gamma_phi_ghz

    if gamma_phi_per_ns == 0:
        return np.inf

    t_phi_ns = 1 / gamma_phi_per_ns
    return t_phi_ns, epsilon_1


def run_fidelity_and_leakage_analysis(params: dict, gate_alpha_factor=4,calculate_population=False):
    """
    Transmon의 Fidelity와 Leakage(L1, L2)를 함께 분석하는 통합 시뮬레이션.
    펄스 모양을 현실적인 가우시안 펄스로 개선.
    게이트 시간을 비조화성(alpha)에 따라 동적으로 계산.

    Args:
        params (dict): EJ, EC, alpha, T1 등 트랜스몬 파라미터 딕셔너리.
        gate_alpha_factor (float): 게이트 시간 계산을 위한 팩터.
                                   gate_time_ns = gate_alpha_factor / alpha_GHz.
                                   값이 클수록 게이트가 길고 스펙트럼이 좁아집니다.
    """
    # --- 1. 파라미터 설정 및 디코히런스 계산 ---
    ej_ghz = params['EJ_GHz']
    ec_ghz = params['EC_MHz'] / 1000.0
    alpha_ghz = params['alpha_MHz'] / 1000.0
    t1_ns = params['T1_us'] * 1000.0 if params['T1_us'] > 0 else np.inf

    # ⭐ 개선: 비조화성(alpha)에 기반하여 게이트 시간 동적 계산
    # 누설을 방지하기 위해, 펄스 지속 시간(게이트 시간)은 비조화성의 역수보다 충분히 길어야 합니다.
    # 게이트 시간이 길수록 주파수 영역에서 펄스 대역폭이 좁아져
    # |1> -> |2> 전이를 유발할 가능성을 줄입니다.
    gate_time_ns = gate_alpha_factor / abs(alpha_ghz)

    t_phi_ej_ns = calculate_tphi_ej_noise_vanharlingen(ej_ghz, ec_ghz)
    t_phi_charge_ns, epsilon_1_val = calculate_tphi_from_charge_dispersion(ej_ghz, ec_ghz)

    gamma_phi_ej = 1 / t_phi_ej_ns if t_phi_ej_ns > 0 and np.isfinite(t_phi_ej_ns) else 0
    gamma_phi_charge = 1 / t_phi_charge_ns if t_phi_charge_ns > 0 and np.isfinite(t_phi_charge_ns) else 0
    total_gamma_phi_per_ns = gamma_phi_ej + gamma_phi_charge
    t_phi_total_ns = 1 / total_gamma_phi_per_ns if total_gamma_phi_per_ns > 0 else np.inf

    # --- 2. QuTiP 시뮬레이션 공통 설정 ---
    N = 3
    w_alpha = 2 * np.pi * alpha_ghz
    a = qt.destroy(N)
    n = qt.num(N)

    # 해밀토니안 (회전 프레임)
    H0 = (w_alpha / 2) * n * (n - 1)
    H_drive = a + a.dag()

    # 가우시안 펄스
    sigma = gate_time_ns / 6.0
    # π 펄스(X 게이트)가 되도록 진폭(A)을 보정
    # ∫ A * exp(...) dt = π/2 (qutip의 H1 계수 때문에 π가 아닌 π/2)
    integral_val, _ = quad(lambda t: np.exp(-(t - gate_time_ns / 2) ** 2 / (2 * sigma ** 2)), 0, gate_time_ns)
    pulse_amp = (np.pi/2.0) / integral_val if integral_val != 0 else 0

    def gaussian_pulse(t, args):
        return pulse_amp * np.exp(-(t - args['t_gate'] / 2) ** 2 / (2 * sigma ** 2))

    H = [H0, [H_drive, gaussian_pulse]]

    tlist = np.linspace(0, gate_time_ns, 151)
    args = {'t_gate': gate_time_ns}
    options = Options(nsteps=50000)

    # Collapse operator 리스트
    c_ops = []
    if t1_ns > 0 and np.isfinite(t1_ns):
        c_ops.append(np.sqrt(1.0 / t1_ns) * a)
    if t_phi_total_ns > 0 and np.isfinite(t_phi_total_ns):
        c_ops.append(np.sqrt(total_gamma_phi_per_ns) * 2 * n)

    # --- 3. 평균 게이트 충실도 계산 ---
    U_ideal_3level = qt.Qobj([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1]], dims=[[3], [3]])

    rho_0_list = [
        qt.fock_dm(2, 0), qt.fock_dm(2, 1),
        (qt.basis(2, 0) + qt.basis(2, 1)).unit().proj(),
        (qt.basis(2, 0) - qt.basis(2, 1)).unit().proj(),
        (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit().proj(),
        (qt.basis(2, 0) - 1j * qt.basis(2, 1)).unit().proj()
    ]
    state_fidelities = []


    for i, rho0_2level in enumerate(rho_0_list):
        # 초기 상태를 3레벨 공간으로 확장
        matrix_3level = np.zeros((3, 3), dtype=complex)
        matrix_3level[:2, :2] = rho0_2level.full()
        rho0_3level = qt.Qobj(matrix_3level, dims=[[3], [3]])

        result = qt.mesolve(H, rho0_3level, tlist, c_ops, args=args, options=options)

        # 시뮬레이션 후의 최종 상태 (3레벨)
        rho_final_3level = result.states[-1]

        # 이상적인 최종 상태 계산 (3레벨 공간에서)
        rho_ideal_final_3level = U_ideal_3level * rho0_3level * U_ideal_3level.dag()

        # 3레벨 전체 공간에서 충실도를 계산하여 누설(leakage) 효과를 반영
        fidelity = qt.fidelity(rho_ideal_final_3level, rho_final_3level)
        state_fidelities.append(fidelity)

    average_fidelity = np.mean(state_fidelities)
    gate_error = 1 - average_fidelity

    # --- 4. 논문 기반 누설률(L1) 및 유입률(L2) 계산 ---
    # L1 계산: 계산 공간의 완전 혼합 상태에서 시작
    rho0_L1 = (qt.fock_dm(N, 0) + qt.fock_dm(N, 1)) / 2.0
    result_L1 = qt.mesolve(H, rho0_L1, tlist, c_ops, args=args, options=options)
    L1_rate = qt.expect(qt.projection(N, 2, 2), result_L1.states[-1])

    # L2 계산: 누설 공간(|2>)에서 시작
    rho0_L2 = qt.fock_dm(N, 2)
    result_L2 = qt.mesolve(H, rho0_L2, tlist, c_ops, args=args, options=options)
    # 계산 공간으로 돌아온 인구 합산
    pop_0 = qt.expect(qt.projection(N, 0, 0), result_L2.states[-1])
    pop_1 = qt.expect(qt.projection(N, 1, 1), result_L2.states[-1])
    L2_rate = pop_0 + pop_1

    # ---  Population Dynamics 계산 ---
    population_data = None
    if calculate_population:
        # |0> 상태에서 시뮬레이션 시작
        rho0_ground = qt.fock_dm(N, 0)
        result_pop = qt.mesolve(H, rho0_ground, tlist, c_ops, args=args, options=options)

        # 프로젝션 연산자를 사용하여 각 상태의 population 계산
        p0_op = qt.projection(N, 0, 0)
        p1_op = qt.projection(N, 1, 1)
        p2_op = qt.projection(N, 2, 2)

        pop0 = qt.expect(p0_op, result_pop.states)
        pop1 = qt.expect(p1_op, result_pop.states)
        pop2 = qt.expect(p2_op, result_pop.states)

        population_data = {
            'tlist': tlist,
            'pop0': pop0,
            'pop1': pop1,
            'pop2': pop2
        }

    # --- 5. 최종 결과 반환 ---
    num_gates = t_phi_total_ns / gate_time_ns if gate_time_ns > 0 and np.isfinite(t_phi_total_ns) else 0

    return ({
        'T_1' : t1_ns,
        'T_phi_ej_us': t_phi_ej_ns / 1000,
        'T_phi_charge_us': t_phi_charge_ns / 1000,
        'T_phi_total_us': t_phi_total_ns / 1000,
        'Charge_dispersion' : epsilon_1_val,
        'Gate_Time_ns': gate_time_ns,
        'Avg_Fidelity': average_fidelity,
        'Gate_Error_%': gate_error * 100,
        'L1_Rate_%': L1_rate * 100,
        'L2_Rate_%': L2_rate * 100,
        'Num_Gates': num_gates
    },population_data)


def plot_results(results_df: pd.DataFrame):
    """
    시뮬레이션 결과를 받아 다양한 관점에서 그래프를 생성하고 저장합니다.
    Gate Error와 Leakage Rate(L1)의 관계를 중점적으로 시각화합니다.
    """
    print("\n[INFO] Generating results plot...")

    # 4x2 레이아웃으로 서브플롯 생성
    fig, axs = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle('Qubit Parameter, Fidelity, and Leakage Analysis vs. Lj at Ec}', fontsize=20, y=0.98)

    # --- 1. Gate Error & L1 Leakage Rate vs. Lj (핵심 Trade-off 그래프) ---
    ax1 = axs[0, 0]
    ax1.plot(results_df['Lj_nH'], results_df['Gate_Error_%'], 'o-', color='crimson', markersize=4,
             label='Total Gate Error')
    ax1.plot(results_df['Lj_nH'], results_df['L1_Rate_%'], 's--', color='darkviolet', markersize=4,
             label='L1 Leakage Rate')
    ax1.set_xlabel('Lj (nH)', fontsize=12)
    ax1.set_ylabel('Error Rate (%)', fontsize=12)
    ax1.set_title('Gate Error & Leakage vs. Lj', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_yscale('log')
    ax1.legend()

    # --- 2. Qubit Frequency vs. Lj ---
    ax2 = axs[0, 1]
    ax2.plot(results_df['Lj_nH'], results_df['f_Q_GHz'], 'o-', color='royalblue', markersize=4)
    ax2.set_xlabel('Lj (nH)', fontsize=12)
    ax2.set_ylabel('Qubit Frequency (GHz)', fontsize=12)
    ax2.set_title('Qubit Frequency', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- 3. Anharmonicity vs. Lj ---
    ax3 = axs[1, 0]
    ax3.plot(results_df['Lj_nH'], results_df['alpha_MHz'], 'o-', color='darkgreen', markersize=4)
    ax3.set_xlabel('Lj (nH)', fontsize=12)
    ax3.set_ylabel('Anharmonicity (MHz)', fontsize=12)
    ax3.set_title('Anharmonicity', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # --- 4. Dephasing Time Limit vs. Lj ---
    ax4 = axs[1, 1]
    ax4.plot(results_df['Lj_nH'], results_df['T_phi_total_us'], 'o-', color='darkorange', markersize=4)
    ax4.set_xlabel('Lj (nH)', fontsize=12)
    ax4.set_ylabel('T_phi Limit (µs)', fontsize=12)
    ax4.set_title('Dephasing Time Limit', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.6)

    # --- 5. Total Gates within Coherence Time vs. Lj ---
    ax5 = axs[2, 0]
    ax5.plot(results_df['Lj_nH'], results_df['Num_Gates'], 'o-', color='purple', markersize=4)
    ax5.set_xlabel('Lj (nH)', fontsize=12)
    ax5.set_ylabel('Number of Gates', fontsize=12)
    ax5.set_title('Total Gates within Coherence Time', fontsize=14)
    ax5.grid(True, linestyle='--', alpha=0.6)
    ax5.set_yscale('log')

    # --- 6. Gate Error & L1 Leakage Rate vs. Anharmonicity ---
    ax6 = axs[2, 1]
    ax6.plot(abs(results_df['alpha_MHz']), results_df['Gate_Error_%'], 'o-', color='darkred', markersize=4,
             label='Total Gate Error')
    ax6.plot(abs(results_df['alpha_MHz']), results_df['L1_Rate_%'], 's--', color='darkviolet', markersize=4,
             label='L1 Leakage Rate')
    ax6.set_xlabel('Anharmonicity |α| (MHz)', fontsize=12)
    ax6.set_ylabel('Error Rate (%)', fontsize=12)
    ax6.set_title('Error Rates vs. Anharmonicity', fontsize=14)
    ax6.grid(True, linestyle='--', alpha=0.6)
    ax6.set_yscale('log')
    ax6.legend()

    # --- 7. Gate Error & L1 Leakage Rate vs. Ej/Ec (새로 추가된 그래프) ---
    ax7 = axs[3, 0]
    ax7.plot(results_df['Ej/Ec'], results_df['Gate_Error_%'], 'o-', color='teal', markersize=4, label='Total Gate Error')
    ax7.plot(results_df['Ej/Ec'], results_df['L1_Rate_%'], 's--', color='orange', markersize=4, label='L1 Leakage Rate')
    ax7.set_xlabel('Ej/Ec Ratio', fontsize=12)
    ax7.set_ylabel('Error Rate (%)', fontsize=12)
    ax7.set_title('Error Rates vs. Ej/Ec Ratio', fontsize=14)
    ax7.grid(True, linestyle='--', alpha=0.6)
    ax7.set_yscale('log')
    ax7.legend()

    ax8 = axs[3, 1]
    ax8.plot(results_df['Ej/Ec'], results_df['Num_Gates'], 'o-', color='teal', markersize=4, label='Number of Gates')
    ax8.set_xlabel('Ej/Ec Ratio', fontsize=12)
    ax8.set_ylabel('Number of Gates', fontsize=12)
    ax8.set_title('Error Rates vs. number of gates', fontsize=14)
    ax8.grid(True, linestyle='--', alpha=0.6)
    ax8.set_yscale('log')
    ax8.legend()

    # 전체 레이아웃 조정 및 파일 저장
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("simulation_results_plot.png", dpi=300)
    print(f"\n[SUCCESS] Plot saved to 'simulation_results_plot.png'.")


def main(final_summary_df):
    """
    Fidelity 및 Leakage 시뮬레이션을 실행하고, 결과를 취합하여
    콘솔 출력, CSV 저장, 그래프 생성을 수행합니다.
    """
    results = []
    total_rows = len(final_summary_df)
    population_plots_to_generate = []

    # DataFrame의 각 행에 대해 시뮬레이션 실행
    for index, row in final_summary_df.iterrows():
        print(f"Processing row {index + 1}/{total_rows}: Lj = {row['Lj_nH']:.4f} nH...")

        lj_val = row['Lj_nH']
        should_calc_pop = (9 <= lj_val <= 15) #9<Lj<15 nH approximatly means 40<EJ<70 GHz

        sim_result, pop_data = run_fidelity_and_leakage_analysis(
            row.to_dict(),
            calculate_population=should_calc_pop
        )
        results.append(sim_result)
        if should_calc_pop and pop_data:
            # plot에 필요한 메타데이터를 함께 딕셔너리로 묶어 저장
            plot_info = pop_data
            plot_info['lj_val'] = lj_val
            plot_info['ec_val'] = row['EC_MHz']
            plot_info['gate_time'] = sim_result['Gate_Time_ns']
            population_plots_to_generate.append(plot_info)

    # 시뮬레이션 결과를 원본 DataFrame과 결합
    results_df = pd.DataFrame(results)
    final_df_full = pd.concat([final_summary_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    # --- 새로 추가된 부분: Ej/Ec 비율 계산 ---
    # EJ는 GHz, EC는 MHz 단위이므로 단위를 맞춰줍니다 (EJ * 1000 / EC).
    final_df_full['Ej/Ec'] = (final_df_full['EJ_GHz'] * 1000) / final_df_full['EC_MHz']

    # 파일명 생성을 위한 메타데이터 추출
    ec_val = final_df_full['EC_MHz'].iloc[0]
    lj_start = final_df_full['Lj_nH'].iloc[0]
    lj_end = final_df_full['Lj_nH'].iloc[-1]

    # 출력할 컬럼 리스트에 Ej/Ec 및 누설 관련 지표 추가
    output_columns = [
        'Lj_nH',
        'Ej/Ec',  # <-- 추가
        'f_Q_GHz',
        'EJ_GHz',
        'EC_MHz',
        'alpha_MHz',
        'gbus1_in_MHz',
        'chi_bus1_MHz',
        'T1_us',
        'T_phi_total_us',
        'Charge_dispersion',
        'Gate_Time_ns',
        'Gate_Error_%',
        'L1_Rate_%',
        'L2_Rate_%',
        'Num_Gates'
    ]

    final_df_display = final_df_full[output_columns]

    # 콘솔에 결과 출력
    print("\n" + "=" * 80)
    print(" " * 20 + "Fidelity and Leakage Simulation Summary")
    print("=" * 80)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 120)
    print(final_df_display)
    print("=" * 80)

    # 결과를 CSV 파일로 저장하고 그래프 그리기
    if not final_df_display.empty:
        # 3. 파일명을 더 명확하게 수정
        csv_filename = f"Ec{ec_val:.2f}_Lj{lj_start:.2f}_{lj_end:.2f}_{pad_gap_num:.2f}_{pad_width_num:.2f}_{pad_height_num:.2f}_{pocket_width_num:.2f}_{pocket_height_num:.2f}fidelity_and_leakage.csv"
        final_df_display.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n[SUCCESS] Results successfully saved to '{csv_filename}'.")
        plot_results(final_df_display)



# --- 반드시 final_summary_df를 생성한 후 실행
if __name__ == '__main__':
    main(final_summary_df)
