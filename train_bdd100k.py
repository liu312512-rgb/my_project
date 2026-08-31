# Monkey-patch: 修复 PTL 2.6.x 中 sized_len 不捕获 RuntimeError 的 bug
# CombinedLoader.__len__() 在 iter() 之前调用会抛出 RuntimeError，
# 但 sized_len 只捕获了 TypeError 和 NotImplementedError
from lightning_fabric.utilities import data as _fabric_data

_original_sized_len = _fabric_data.sized_len


def _patched_sized_len(dataloader):
    try:
        return _original_sized_len(dataloader)
    except RuntimeError:
        return None


_fabric_data.sized_len = _patched_sized_len

# ---------------------------------------------------------------------------
# 运行方式（重要！）：
#   uv run python train_bdd100k.py
# 不要直接用 `python train_bdd100k.py`，否则会使用 conda 环境的旧版 PyTorch
# ---------------------------------------------------------------------------
import sys
import torch


def _check_gpu():
    """验证 PyTorch 版本是否支持当前 GPU（Blackwell sm_120 需要 cu130+）"""
    if not torch.cuda.is_available():
        sys.exit(
            "错误: CUDA PyTorch 不可用。\n"
            "请用 `uv run python train_bdd100k.py` 运行，不要用 `python train_bdd100k.py`。"
        )
    major, minor = torch.cuda.get_device_capability()
    if major >= 12 and "+cu" in torch.__version__:
        cu_ver = torch.__version__.split("+cu")[1]
        if int(cu_ver) < 130:
            sys.exit(
                f"错误: PyTorch {torch.__version__} 不支持 Blackwell GPU (sm_{major}{minor})。\n"
                f"当前使用: {sys.executable}\n"
                f"请用 `uv run python train_bdd100k.py` 运行。"
            )
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__} | OK")


_check_gpu()

from rfdetr import RFDETRBase

if __name__ == '__main__':

    model = RFDETRBase(num_classes=10)

    model.train(
        dataset_dir="D:/RF-DETR/rf-detr-develop/BDD100K_coco",
        dataset_file="roboflow",
        epochs=100,
        batch_size=4,
        lr=1e-4,
        num_workers=2,
        output_dir="output/bdd100k",
        device="cuda",
        progress_bar="rich",  # 显示每个 epoch 的训练进度条
    )
