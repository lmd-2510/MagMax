# MagMax baseline trên MICA

Giữ nguyên source training của tác giả. Chỉ thêm `merge_magmax_only.py` dùng lại
`TaskVector`, `merge_max_abs` và `eval_single_dataset` gốc, cùng bash quản lý job.
Một GPU vật lý 0 hoặc 1, một lần chạy với seed=5.

## Setup trên server

### Dùng ai_env hiện có (theo lựa chọn của bạn)

Sau khi clone repo ở `~/DrugDrug/MagMax`, chạy:

```bash
cd ~/DrugDrug/MagMax
git pull --ff-only
conda activate ai_env
mkdir -p outs
python - <<'PY' > outs/torch_constraints.txt
from importlib.metadata import version, PackageNotFoundError
for name in ('torch', 'torchvision', 'torchaudio'):
    try:
        print(f'{name}=={version(name)}')
    except PackageNotFoundError:
        if name != 'torchaudio':
            raise
PY
python -m pip install -r requirements-magmax.txt -c outs/torch_constraints.txt
```

Constraints giữ nguyên bản torch/torchvision/torchaudio đang có, tránh pip tự đổi bản CUDA.
Các package khác trong requirements được cài/đổi về phiên bản ghi trong file.
Launcher kiểm tra import và CUDA thực tế, chấp nhận runtime ai_env khác export gốc;
hyperparameters training giữ nguyên. Phiên bản runtime được lưu cùng output
(`environment.txt`, `runtime.txt`); nếu khác thì không coi là tái lập nguyên môi trường tác giả.
Python 3.9-3.12 được chấp nhận cho bộ dependency đã ghim; bản gốc là 3.10.14.
Launcher đặt `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` để code load full-model checkpoint
gốc hoạt động với PyTorch >=2.6, theo tài liệu:
https://docs.pytorch.org/docs/2.6/notes/serialization.html#environment-variables

### Tùy chọn môi trường nguyên bản

Nếu chưa có repo:

```bash
mkdir -p ~/DrugDrug
cd ~/DrugDrug
git clone https://github.com/lmd-2510/MagMax.git
cd MagMax
conda env create -f environment.yml
conda activate magmax
```

Nếu repo đã có, `cd` vào repo và `git pull --ff-only`. Nếu môi trường `magmax` đã tồn tại,
activate rồi chạy `check`; không chạy lại `conda env create` với cùng tên.
File môi trường nguyên bản dành cho Linux; đây là môi trường riêng với `ai_env` của TDDI.

## Chọn GPU và chạy

```bash
cd ~/DrugDrug/MagMax
conda activate ai_env
nvidia-smi
export MAGMAX_GPU_ID=0
bash scripts/run_magmax_mica.sh check
bash scripts/run_magmax_mica.sh start
```

Đổi thành `MAGMAX_GPU_ID=1` nếu GPU 1 là GPU bạn được dùng. Chỉ chọn một GPU;
không đặt `0,1`. Kiểm tra lượng VRAM trống và job đang dùng ở thời điểm chạy.
`check` không train/download, hiển thị môi trường, GPU/VRAM và đĩa trống.
Khuyến nghị 32 GB RAM, khoảng 16 vCPU (source gốc dùng 16 workers), 30 GB đĩa trống.
Lần chạy đầu tự tải CIFAR-100 và OpenAI CLIP, cần Internet.

Sau `[STARTED]`, job chạy bằng nohup; có thể đóng SSH hoặc máy Windows.
Server phải tiếp tục hoạt động. Xem log/kết quả:

```bash
bash scripts/run_magmax_mica.sh status
bash scripts/run_magmax_mica.sh follow
cat outs/magmax_CIFAR100_20_seed5/result.json
```

Ctrl+C trong `follow` chỉ dừng xem log. Chỉ coi job hoàn tất khi `status` báo
`exit_code=0` và có kết quả `result.json` từ lần chạy đó.

## Recipe

- CIFAR-100: 50.000 train / 10.000 test, class order gốc, 20 task × 5 lớp.
- OpenAI CLIP ViT-B/16 (`ViT-B-16`), sequential fine-tuning từ task trước.
- 10 epoch/task, batch128, FP32; không AMP/gradient accumulation/replay.
- AdamW LR1e-5, weight decay0.1, defaults PyTorch2.4.0; reset mỗi task.
- CrossEntropy, label smoothing0, gradient clip norm1.0; head/text freeze theo source gốc.
- Warmup500 reset mỗi task; 200 steps/task nằm hoàn toàn trong warmup, giữ nguyên.
- Merge signed max-absolute delta so với theta0, lambda0.5, task sau thắng khi bằng độ lớn.
- Top-1 final model trên toàn test, head100class, `val_preprocess` bằng evaluator gốc.
- Không chạy RandMix/averaging/TIES, không WandB và không quét lambda.

Output nằm trong `outs/magmax_CIFAR100_20_seed5/`: `result.json` có `top1_percent`,
model merged, log, PID, exit code, Git commit, training command và thời gian.
Giữ `checkpoints/ViT-B-16/zeroshot.pt` cùng 20 checkpoint ở:
`checkpoints/ViT-B-16/sequential_finetuning/class_incremental/CIFAR100-20/ft-epochs-10-seed:5/`.
Chúng và output/datasets đã được gitignore; không push file lớn lên GitHub.

Để chỉ merge/evaluate khi đã đủ checkpoint:

```bash
bash scripts/run_magmax_mica.sh merge
```

Script không tự xóa checkpoint hoặc tự resume training bị ngắt. Code gốc không lưu
RNG/optimizer state để resume đúng run liên tục. Nếu chỉ có một phần checkpoint,
giữ/đổi tên thư mục task checkpoint trước khi chạy fresh; nếu đủ 20, chúng được tái sử dụng
và bạn cần chắc chúng đến từ cùng config. Khi lần launch trước đã có exit code,
`start` yêu cầu kiểm tra trạng thái trước, tránh âm thầm ghi đè kết quả.

## Chạy lại sau lỗi tải CIFAR100, chưa có checkpoint task

Nếu lần trước lỗi integrity của dataset trước khi train và Tunable MagMax đang
tải/giải nén dataset dùng chung, có thể đặt MagMax vào hàng chờ ngay:

```bash
cd ~/DrugDrug/MagMax
conda activate ai_env
git pull --ff-only
export MAGMAX_GPU_ID=0
export MAGMAX_DATA_ROOT="$PWD/datasets"
bash scripts/run_magmax_mica.sh retry
bash scripts/run_magmax_mica.sh follow
```

`retry` chỉ chấp nhận lần chạy đã thất bại và không có checkpoint task nào.
Di chuyển file log/exit code/metadata cũ vào `outs/magmax_CIFAR100_20_seed5/failed_attempts/`
để giữ lịch sử; không xóa dataset, pretrained cache, zeroshot hoặc task checkpoints.
Job mới chạy nền, hiện `[STAGE] waiting_dataset`, không download thêm, không sửa
training/config. Kiểm tra MD5 train/test/meta mỗi 30 giây; đủ dữ liệu thì tự train
trên GPU0 và merge/evaluate. Nếu bước dataset của Tunable báo lỗi, job chờ dừng
và yêu cầu xem log đó. Thời gian chờ không nằm trong `total_seconds.txt` training/merge.

PID chờ lưu trong queue.pid; job.pid chỉ được tạo khi bắt đầu training, để
Tunable không hiểu nhầm job chờ này là một job đang download và chờ ngược lại.
Khóa launcher giữ xuyên suốt thời gian chờ và training. Dùng `status` như trước;
sau `[STARTED]` có thể đóng SSH/máy cá nhân. Không chạy lại `retry` khi job còn chạy.

Nguồn recipe: `scripts/CIL/all.sh`, `src/args.py`, `finetune_splitted.py`,
`src/utils.py`, `merge_splitted.py`, `src/merging/task_vectors.py` và `src/eval.py`
tại upstream commit `4f49f18fb30216d3a0dbe4220b469a3273e45475`.
Chưa train trên server của bạn; chưa có accuracy hoặc thời gian thực tế của RTX4090.
