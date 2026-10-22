# Phase 4 環境確認

監査日：2026-10-21 JST。対象HEAD `5b8c7f5251128843b313b88304c3f2d4261e1cf0`、branch `main`。実行環境はLinux aarch64、kernel `7.0.0-1019-nvidia`。RTX 4090/A6000が利用できるという前提は置いていない。

**FAIL：このセッションから利用可能なCUDA GPUは0枚。GPU実測は未実施。**

| 確認項目 | 実測した状態 |
| --- | --- |
| nvidia-smi | 終了コード9、NVIDIA driverと通信できない |
| NVIDIAデバイスノード | `/dev/nvidia*` は存在しない |
| NVIDIA kernel driver公開情報 | `/proc/driver/nvidia/version` は存在しない |
| kernel module | lsmodでnvidia/nouveauの行なし（検索終了コード1） |
| PCI上の表示装置 | `000f:01:00.0`、NVIDIA、PCI ID `10de:2e12`、VGA controller 1件を観測 |
| GPUの商品名・実枚数・VRAM総量/空き | ドライバ経由で確認不能。PCI表示1件をCUDA GPU1枚として数えていない |
| PyTorch CUDAビルド | torch 2.10.0+cu128、CUDA 12.8、cuDNN 91002 |
| PyTorch認識 | is_available=False、device_count=0 |
| CUDA明示初期化 | `torch.cuda.init()` → RuntimeError: No CUDA GPUs are available |
| CPU | 20 logical CPU、ARM Cortex-X925 / Cortex-A725（lscpu出力を保存） |
| RAM | 128,517,992,448 byte（約119.7 GiB）、MemAvailable 123,860,340,736 byte（約115.4 GiB） |
| Swap | 約16 GiB、観測時使用0 |
| 保存先 | `/home/nakagawa` と `/tmp` は同じ `/dev/nvme0n1p2` |
| 空き容量 | 取得時約3,087.108 GB（2,875.094 GiB） |
| Python | `/home/nakagawa/venvs/swallow-py312/bin/python`、Python 3.12.3 |

CUDA対応PyTorchがimportできても、CUDA deviceが利用可能とは限らない。ドライバ未稼働・未公開状態の原因の特定や、ホスト側の復旧は本監査の実施範囲に含めていない。ドライバ導入・moduleロード・再起動・依存パッケージ更新は行っていない。

## 既存ライブラリ

| モジュール | import | 実バージョン |
| --- | --- | --- |
| torch | PASS | 2.10.0+cu128 |
| transformers | PASS | 5.8.0 |
| pytorch_lightning | PASS | 2.2.1 |
| huggingface_hub | PASS | 1.14.0 |
| numpy | PASS | 1.26.4 |
| sklearn | PASS | 1.6.1 |
| iterstrat | PASS | 0.1.9 |
| tokenizers | PASS | 0.22.2 |
| safetensors | PASS | 0.5.3 |

requirements.txtはバージョンを固定していないため、今回の実importバージョンを再現情報として保存した。追加インストールはしていない。システムPythonへの切替や別GPU環境の推測もしていない。

## BERTキャッシュ

`/home/nakagawa/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594` にconfig、vocab、tokenizer、model.safetensors（440,449,768 byte）が存在する。`snapshot_download(..., local_files_only=True)` のローカル解決だけを実行した。今回BERTをGPUにロードできたという意味ではない。HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、PYTHONDONTWRITEBYTECODE=1を使用し、モデルダウンロードは行っていない。

## 既存GPU設定・出力先

Phase 3 `training.main` の `--device` の既定値はcpuで、GPU指定は `--device cuda:0`。実学習はbatch_size=8、max_length=512、FP32、autocast/GradScalerなし。`torch.use_deterministic_algorithms(True)` を使用し、CUDA起動時は `CUBLAS_WORKSPACE_CONFIG=:4096:8` が前提。旧Lightning学習コードはaccelerator=gpu・devices=1、batch8で、明示的なmixed precision設定はない。matmul_precision(high)はmixed precisionの有効化とは区別した。

Phase 3保存先は必須の--output引数で指定し、既存ディレクトリなら停止する。checkpointは `best_ratio_<r>_<fold>_lr_<lr>.pth`、最終重みはmodel_weights.pth、ラベルJSON、run_manifest.json、training_results.jsonを保存する。永続的なコンソールログ収集は外側で行う必要がある。今回の専用出力先は本ディレクトリのみで、学習checkpointは新規作成していない。

生ログ：logs/nvidia_smi.log、logs/nvidia_devices.log、logs/kernel_driver.log、logs/modules.log、logs/gpu_pci.log、logs/python_environment.json、logs/cpu.log、logs/memory.log、logs/disk.log。
