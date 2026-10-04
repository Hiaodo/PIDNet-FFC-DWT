"""
precompute_edges.py
-------------------
离线预计算 Cityscapes / CamVid 数据集的边缘图，避免训练时每步执行 Canny+dilate。

逻辑与 BaseDataset.gen_sample() 前半段完全一致：
  1. 读取 label（灰度图）
  2. convert_label（对 Cityscapes 做 id 映射）
  3. cv2.Canny(label, 0.1, 0.2)
  4. edge_pad（去除 y_k_size/x_k_size 边框后用 0 补回）
  5. cv2.dilate(kernel=edge_size×edge_size, iterations=1) 并二值化 (>50)*255

输出：与 label 文件同目录、同名，后缀改为 _edge.png（uint8, 0 or 255）

用法示例（从 PIDNet_data 根目录运行）：
  python tools/precompute_edges.py --dataset cityscapes --data-root data/ --edge-size 4

参数说明：
  --dataset    cityscapes 或 camvid
  --data-root  数据根目录（与 yaml 中 DATASET.ROOT 一致）
  --list-dir   list 文件目录前缀，默认 list/<dataset>/
  --edge-size  膨胀核大小，默认 4（与 bd_dilate_size 一致）
  --workers    并行进程数，默认 cpu_count // 2
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm

# ── 与 base_dataset.py 保持完全一致 ─────────────────────────────────────────
Y_K_SIZE = 6
X_K_SIZE = 6

# Cityscapes label id → train id 映射（与 cityscapes.py 一致）
CITYSCAPES_LABEL_MAPPING = {
    -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255,
    5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255,
    11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255,
    17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    29: 255, 30: 255, 31: 16, 32: 17, 33: 18
}

def _build_lut(ignore_label=255):
    lut = np.full(256, ignore_label, dtype=np.uint8)
    for k, v in CITYSCAPES_LABEL_MAPPING.items():
        if 0 <= k < 256:
            lut[k] = v
    return lut


def compute_edge(label_gray: np.ndarray, edge_size: int, ignore_label: int = 255) -> np.ndarray:
    """
    根据 label（已 convert 后的 train id 图）计算边缘图。
    返回 uint8 数组，值为 0 或 255。
    """
    edge = cv2.Canny(label_gray, 0.1, 0.2)

    # edge_pad（与 gen_sample 中 edge_pad=True 的默认行为一致）
    edge_inner = edge[Y_K_SIZE:-Y_K_SIZE, X_K_SIZE:-X_K_SIZE]
    edge = np.pad(edge_inner, ((Y_K_SIZE, Y_K_SIZE), (X_K_SIZE, X_K_SIZE)), mode='constant')

    kernel = np.ones((edge_size, edge_size), np.uint8)
    edge_dilated = cv2.dilate(edge, kernel, iterations=1)

    # 二值化：>50 → 255，其余 → 0（保存为 uint8）
    edge_bin = (edge_dilated > 50).astype(np.uint8) * 255
    return edge_bin


# ── 各 dataset 的处理入口 ────────────────────────────────────────────────────

class CityscapesEdgeWorker:
    """封装为可 pickle 的类，供 multiprocessing.Pool.imap 使用"""
    def __init__(self, data_root: str, edge_size: int):
        self.root = Path(data_root) / 'cityscapes'
        self.edge_size = edge_size
        self.lut = _build_lut()

    def __call__(self, label_rel: str):
        label_path = self.root / label_rel
        label_gray = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label_gray is None:
            return f"[WARN] 读取失败: {label_path}"

        # convert_label（与 cityscapes.py 一致）
        label_mapped = self.lut[label_gray]

        edge = compute_edge(label_mapped, self.edge_size)

        # 保存为同目录下的 _edge.png
        edge_path = label_path.parent / (label_path.stem + '_edge.png')
        cv2.imwrite(str(edge_path), edge)
        return None   # None 表示成功


class CamvidEdgeWorker:
    """CamVid 不需要 label id 映射，直接对灰度 label 计算 edge"""
    def __init__(self, data_root: str, edge_size: int):
        self.root = Path(data_root) / 'camvid'
        self.edge_size = edge_size

    def __call__(self, label_rel: str):
        label_path = self.root / label_rel
        label_gray = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label_gray is None:
            return f"[WARN] 读取失败: {label_path}"

        edge = compute_edge(label_gray, self.edge_size)
        edge_path = label_path.parent / (label_path.stem + '_edge.png')
        cv2.imwrite(str(edge_path), edge)
        return None


# ── 收集 label 文件列表 ───────────────────────────────────────────────────────

def collect_cityscapes_labels(data_root: str, list_dir: str) -> list:
    """读取 train.lst 和 val.lst，收集所有 label 相对路径"""
    label_paths = set()
    for lst_name in ('train.lst', 'val.lst', 'trainval.lst'):
        lst_file = Path(list_dir) / lst_name
        if not lst_file.exists():
            continue
        with open(lst_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label_paths.add(parts[1])   # 第二列是 label 路径
    return sorted(label_paths)


def collect_camvid_labels(data_root: str, list_dir: str) -> list:
    label_paths = set()
    for lst_name in ('train.lst', 'val.lst', 'test.lst'):
        lst_file = Path(list_dir) / lst_name
        if not lst_file.exists():
            continue
        with open(lst_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label_paths.add(parts[1])
    return sorted(label_paths)


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='离线预计算 Canny edge 并保存为 PNG 文件'
    )
    parser.add_argument('--dataset', default='cityscapes',
                        choices=['cityscapes', 'camvid'],
                        help='数据集名称')
    parser.add_argument('--data-root', default='data/',
                        help='数据根目录（与 yaml 中 DATASET.ROOT 一致）')
    parser.add_argument('--list-dir', default=None,
                        help='lst 文件所在目录，默认为 data/list/<dataset>/')
    parser.add_argument('--edge-size', type=int, default=4,
                        help='膨胀核大小（与 bd_dilate_size 一致，默认 4）')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() // 2),
                        help='并行进程数')
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = args.data_root
    list_dir = args.list_dir or os.path.join(data_root, 'list', args.dataset)

    print(f"[INFO] 数据集  : {args.dataset}")
    print(f"[INFO] 数据根目录: {data_root}")
    print(f"[INFO] lst 目录 : {list_dir}")
    print(f"[INFO] edge_size: {args.edge_size}")
    print(f"[INFO] workers  : {args.workers}")

    if args.dataset == 'cityscapes':
        label_list = collect_cityscapes_labels(data_root, list_dir)
        worker = CityscapesEdgeWorker(data_root, args.edge_size)
    else:
        label_list = collect_camvid_labels(data_root, list_dir)
        worker = CamvidEdgeWorker(data_root, args.edge_size)

    if not label_list:
        print(f"[ERROR] 未找到任何 label 文件，请检查 --list-dir 路径: {list_dir}")
        sys.exit(1)

    print(f"[INFO] 共找到 {len(label_list)} 个 label 文件，开始处理…")

    errors = []
    if args.workers <= 1:
        for lbl in tqdm(label_list, unit='img'):
            err = worker(lbl)
            if err:
                errors.append(err)
    else:
        with Pool(processes=args.workers) as pool:
            for err in tqdm(pool.imap_unordered(worker, label_list),
                            total=len(label_list), unit='img'):
                if err:
                    errors.append(err)

    print(f"\n[INFO] 完成！成功: {len(label_list) - len(errors)}，失败: {len(errors)}")
    for e in errors:
        print(e)


if __name__ == '__main__':
    main()
