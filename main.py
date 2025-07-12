#!/usr/bin/env python3
"""
스낵 탐지 모델 전체 파이프라인 실행기
실행 순서: 데이터 분할 → 모델 훈련 → OpenVINO 변환 → INT8 양자화
"""

import os
import sys
from pathlib import Path
import argparse

# 로컬 모듈 임포트
from split_dataset_01 import split_dataset
from make_model_02 import train_custom_yolov8, create_data_yaml
from custom_model_quantization_03 import OpenVINOConverter
from conver_int8_04 import INT8Converter


class SnackDetectionPipeline:
    """스낵 탐지 모델 전체 파이프라인"""

    def __init__(self, source_dataset_path, output_base_path="./pipeline_output"):
        """
        Args:
            source_dataset_path (str): 원본 데이터셋 경로 (images/, labels/ 포함)
            output_base_path (str): 파이프라인 출력 기본 경로
        """
        self.source_dataset = Path(source_dataset_path)
        self.output_base = Path(output_base_path)
        self.output_base.mkdir(exist_ok=True)

        # 각 단계별 경로 설정
        self.split_data_path = self.output_base / "split_dataset"
        self.model_output_path = self.output_base / "trained_model"
        self.openvino_path = None
        self.int8_path = None

        print(f"🚀 스낵 탐지 파이프라인 초기화")
        print(f"📂 원본 데이터: {self.source_dataset}")
        print(f"📁 출력 디렉토리: {self.output_base}")

    def step1_split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """1단계: 데이터셋 분할"""
        print("\n" + "=" * 60)
        print("1️⃣ 데이터셋 분할 시작")
        print("=" * 60)

        try:
            train_files, val_files, test_files = split_dataset(
                source_dir=str(self.source_dataset),
                output_dir=str(self.split_data_path),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )

            if train_files and val_files and test_files:
                print("✅ 1단계 완료: 데이터셋 분할 성공")
                return True
            else:
                print("❌ 1단계 실패: 데이터셋 분할 실패")
                return False

        except Exception as e:
            print(f"❌ 1단계 오류: {e}")
            return False

    def step2_train_model(self, class_names, epochs=100, model_size='s'):
        """2단계: YOLOv8 모델 훈련"""
        print("\n" + "=" * 60)
        print("2️⃣ YOLOv8 모델 훈련 시작")
        print("=" * 60)

        try:
            # data.yaml 생성
            data_yaml_path = create_data_yaml(
                dataset_path=str(self.split_data_path),
                class_names=class_names,
                output_path="data.yaml"
            )

            # 모델 훈련
            model, results = train_custom_yolov8(
                data_yaml_path=data_yaml_path,
                model_size=model_size,
                epochs=epochs,
                imgsz=640
            )

            # 훈련된 모델 경로 설정
            self.trained_model_path = Path("snack_detection/yolov8s_custom/weights/best.pt")

            if self.trained_model_path.exists():
                print("✅ 2단계 완료: 모델 훈련 성공")
                return True
            else:
                print("❌ 2단계 실패: 훈련된 모델을 찾을 수 없음")
                return False

        except Exception as e:
            print(f"❌ 2단계 오류: {e}")
            return False

    def step3_convert_openvino(self):
        """3단계: OpenVINO 변환 (FP16)"""
        print("\n" + "=" * 60)
        print("3️⃣ OpenVINO 변환 시작 (FP16)")
        print("=" * 60)

        try:
            converter = OpenVINOConverter(str(self.trained_model_path))
            self.openvino_path = converter.get_model_path()

            if self.openvino_path:
                print("✅ 3단계 완료: OpenVINO 변환 성공")
                return True
            else:
                print("❌ 3단계 실패: OpenVINO 변환 실패")
                return False

        except Exception as e:
            print(f"❌ 3단계 오류: {e}")
            return False

    def step4_convert_int8(self):
        """4단계: INT8 양자화 변환"""
        print("\n" + "=" * 60)
        print("4️⃣ INT8 양자화 변환 시작")
        print("=" * 60)

        try:
            converter = INT8Converter(
                model_path=str(self.trained_model_path),
                data_yaml_path="data.yaml"
            )
            self.int8_path = converter.get_model_path()

            if self.int8_path:
                print("✅ 4단계 완료: INT8 양자화 성공")
                return True
            else:
                print("❌ 4단계 실패: INT8 양자화 실패")
                return False

        except Exception as e:
            print(f"❌ 4단계 오류: {e}")
            return False

    def run_full_pipeline(self, class_names, epochs=100, model_size='s'):
        """전체 파이프라인 실행"""
        print("\n🔥 스낵 탐지 모델 전체 파이프라인 시작!")
        print(f"📊 클래스 수: {len(class_names)}")
        print(f"🔄 에포크: {epochs}")
        print(f"📦 모델 크기: YOLOv8{model_size}")

        # 1단계: 데이터셋 분할
        if not self.step1_split_dataset():
            return False

        # 2단계: 모델 훈련
        if not self.step2_train_model(class_names, epochs, model_size):
            return False

        # 3단계: OpenVINO 변환
        if not self.step3_convert_openvino():
            return False

        # 4단계: INT8 양자화
        if not self.step4_convert_int8():
            return False

        # 최종 결과 출력
        self.show_final_results()
        return True

    def _show_model_sizes(self):
        """모델 크기 비교 표시"""
        try:
            if self.trained_model_path and self.trained_model_path.exists():
                pytorch_size = self.trained_model_path.stat().st_size / (1024 * 1024)
                print(f"\n📏 모델 크기 비교:")
                print(f"  🔸 PyTorch (.pt): {pytorch_size:.1f}MB")

                # FP16 모델 크기
                if self.openvino_path:
                    fp16_dir = Path(self.openvino_path).parent
                    if fp16_dir.exists():
                        fp16_size = sum(f.stat().st_size for f in fp16_dir.iterdir() if f.is_file()) / (1024 * 1024)
                        print(f"  🔸 OpenVINO FP16: {fp16_size:.1f}MB ({fp16_size / pytorch_size:.1f}x)")

                # INT8 모델 크기
                if self.int8_path:
                    int8_dir = Path(self.int8_path).parent
                    if int8_dir.exists():
                        int8_size = sum(f.stat().st_size for f in int8_dir.iterdir() if f.is_file()) / (1024 * 1024)
                        print(f"  🔢 OpenVINO INT8: {int8_size:.1f}MB ({int8_size / pytorch_size:.1f}x)")
        except Exception as e:
            print(f"⚠️ 크기 비교 중 오류: {e}")

    def show_final_results(self):
        """최종 결과 요약"""
        print("\n" + "🎉" * 20)
        print("🎉 전체 파이프라인 완료!")
        print("🎉" * 20)

        print(f"\n📋 생성된 모델들:")
        print(f"  🔸 원본 PyTorch: {self.trained_model_path}")
        print(f"  🔸 OpenVINO FP16: {self.openvino_path}")
        print(f"  🔸 OpenVINO INT8: {self.int8_path}")

        print(f"\n📁 파일 위치:")
        print(f"  📂 분할된 데이터: {self.split_data_path}")
        print(f"  📂 훈련 결과: snack_detection/yolov8s_custom/")
        print(f"  📂 OpenVINO 모델: models/")

        print(f"\n🚀 라즈베리파이 배포:")
        print(f"  💡 FP16 모델 사용: {self.openvino_path}")
        print(f"  🔢 INT8 모델 사용: {self.int8_path}")

        # 파일 크기 비교
        self._show_model_sizes()


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='스낵 탐지 모델 전체 파이프라인')

    # 필수 인자
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='원본 데이터셋 경로 (예: ./snack_dataset/train/)')

    # 선택적 인자
    parser.add_argument('--output', '-o', type=str, default='./pipeline_output',
                        help='출력 디렉토리 (기본값: ./pipeline_output)')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='훈련 에포크 수 (기본값: 100)')
    parser.add_argument('--model-size', '-m', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 모델 크기 (기본값: s)')

    args = parser.parse_args()

    # 데이터셋 경로 확인
    if not Path(args.source).exists():
        print(f"❌ 데이터셋 경로가 존재하지 않습니다: {args.source}")
        sys.exit(1)

    # 클래스 이름 정의 (실제 데이터에 맞게 수정 필요)
    class_names = [
        'crown_BigPie_Strawberry', 'crown_ChocoHaim', 'crown_Concho', 'crown_Potto_Cheese_Tart',
        'haetae_Guun_Gamja', 'haetae_HoneyButterChip', 'haetae_Masdongsan',
        'haetae_Osajjeu', 'haetae_Oyeseu', 'lotte_kkokkalkon_gosohanmas',
        'nongshim_Alsaeuchip', 'nongshim_Banana_Kick', 'nongshim_ChipPotato_Original',
        'nongshim_Ojingeojip', 'orion_Chocolate_Chip_Cookies', 'orion_Diget_Choco',
        'orion_Diget_tongmil', 'orion_Fresh_Berry', 'orion_Gosomi',
        'orion_Pocachip_Original', 'orion_chokchokhan_Chocochip'
    ]

    # 파이프라인 실행
    pipeline = SnackDetectionPipeline(
        source_dataset_path=args.source,
        output_base_path=args.output
    )

    success = pipeline.run_full_pipeline(
        class_names=class_names,
        epochs=args.epochs,
        model_size=args.model_size
    )

    if success:
        print("\n✅ 모든 과정이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n❌ 파이프라인 실행 중 오류가 발생했습니다!")
        sys.exit(1)


if __name__ == "__main__":
    main()