import os
import shutil
import random
from pathlib import Path


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    YOLO 형식 데이터셋을 train/val/test로 분할
    
    snack_data_set 필요 URL : https://universe.roboflow.com/korea-nazarene-university/-d9kpq/dataset/3

    Args:
        source_dir: 원본 데이터 디렉토리 (images/, labels/ 폴더 포함)
        output_dir: 분할된 데이터를 저장할 디렉토리
        train_ratio: 훈련 데이터 비율 (기본값: 0.7)
        val_ratio: 검증 데이터 비율 (기본값: 0.2)
        test_ratio: 테스트 데이터 비율 (기본값: 0.1)
    """

    # 비율 검증
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합이 1이 되어야 합니다"

    # 경로 설정 (전달받은 source_dir 사용)
    source_images = Path(source_dir) / "images"
    source_labels = Path(source_dir) / "labels"
    output_path = Path(output_dir)

    # 출력 디렉토리 생성
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 이미지 파일 목록 가져오기
    image_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png"))
    image_files = [f.stem for f in image_files]  # 확장자 제거

    # 무작위 셔플
    random.seed(42)  # 재현 가능한 결과를 위한 시드 설정
    random.shuffle(image_files)

    # 분할 인덱스 계산
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # 데이터 분할
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    # 파일 복사 함수
    def copy_files(file_list, split_name):
        for file_stem in file_list:
            # 이미지 파일 복사
            for ext in ['.jpg', '.png', '.jpeg']:
                src_img = source_images / f"{file_stem}{ext}"
                if src_img.exists():
                    dst_img = output_path / split_name / 'images' / f"{file_stem}{ext}"
                    shutil.copy2(src_img, dst_img)
                    break

            # 라벨 파일 복사
            src_label = source_labels / f"{file_stem}.txt"
            if src_label.exists():
                dst_label = output_path / split_name / 'labels' / f"{file_stem}.txt"
                shutil.copy2(src_label, dst_label)

    # 각 분할에 파일 복사
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    # 결과 출력
    print(f"데이터셋 분할 완료!")
    print(f"Train: {len(train_files)} 개 ({len(train_files) / n_total * 100:.1f}%)")
    print(f"Val: {len(val_files)} 개 ({len(val_files) / n_total * 100:.1f}%)")
    print(f"Test: {len(test_files)} 개 ({len(test_files) / n_total * 100:.1f}%)")

    return train_files, val_files, test_files


if __name__ == "__main__":
    source_directory = "./snack_dataset/train/"  # 원본 데이터셋 경로 (images/, labels/ 폴더 포함)
    output_directory = "./split_snack_data/"  # 분할된 데이터셋 저장 경로, train, val, test 폴더 생성

    # 데이터셋 분할 실행
    train_files, val_files, test_files = split_dataset(
        source_dir=source_directory,
        output_dir=output_directory,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )