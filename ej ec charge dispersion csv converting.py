import os
import glob
import pandas as pd

# ------------------------------------------------------------------
# 사용자 설정 구간
# ------------------------------------------------------------------
# 1. 원본 CSV 파일들이 있는 폴더 경로를 지정하세요.
source_folder = r'C:\Users\User\PycharmProjects\QuTiP'  # 예: r'C:\MyData'

# 2. 결과물을 저장할 파일 이름을 지정하세요.
output_file_name = 'consolidated_data_by_column_Ej_Ec_chargedispersion.csv'
# ------------------------------------------------------------------


def process_and_combine_csvs(input_dir, output_file):
    """
    기존의 파일 탐색 로직을 사용하여 각 CSV의 첫 세 열을
    'Ej', 'Ec', 'Gate Error'로 매핑하고 하나의 CSV로 합칩니다.
    """
    # 기존 코드와 동일한 방식으로 CSV 파일 목록을 가져옵니다.
    csv_pattern = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"❌ 오류: '{input_dir}' 폴더에서 CSV 파일을 찾을 수 없습니다.")
        return

    print(f"📁 총 {len(csv_files)}개의 CSV 파일을 찾았습니다. 데이터 병합을 시작합니다...")

    # 처리된 데이터를 담을 리스트
    data_frames_to_combine = []

    # 새로운 열의 제목(헤더)
    column_map = {0: 'Ec', 1: 'Ej', 2: 'charge dispersion'}

    for file_path in csv_files:
        try:
            # CSV 파일을 헤더 없이 읽어옵니다.
            df = pd.read_csv(file_path, header=None)

            # 파일에 최소 3개의 열이 있는지 확인
            if df.shape[1] < 3:
                print(f"⚠️ 경고: '{os.path.basename(file_path)}' 파일에 열이 3개 미만이라 건너뜁니다.")
                continue

            # 첫 3개의 열만 선택합니다.
            temp_df = df.iloc[:, [4, 3, 10]]

            # 열 이름을 'Ej', 'Ec', 'Gate Error'로 변경합니다.
            temp_df = temp_df.rename(columns=column_map)

            # 리스트에 완성된 데이터프레임을 추가합니다.
            data_frames_to_combine.append(temp_df)

        except Exception as e:
            print(f"❌ 오류: '{os.path.basename(file_path)}' 파일 처리 중 문제가 발생했습니다: {e}")

    if not data_frames_to_combine:
        print("처리할 유효한 데이터가 없습니다. 작업을 중단합니다.")
        return

    # 모든 데이터프레임을 하나로 합칩니다.
    final_df = pd.concat(data_frames_to_combine, ignore_index=True)

    # 결과를 새로운 CSV 파일로 저장합니다.
    final_df.to_csv(output_file, index=False)

    print("-" * 40)
    print(f"✅ 작업 완료! 총 {len(final_df)}개의 행이 처리되었습니다.")
    print(f"결과가 '{output_file}' 파일에 저장되었습니다.")
    print("-" * 40)


if __name__ == '__main__':
    process_and_combine_csvs(source_folder, output_file_name)