# MagMax baseline trên MICA

Giữ nguyên source training của tác giả. Chỉ thêm `merge_magmax_only.py` dùng lại
`TaskVector`, `merge_max_abs` và `eval_single_dataset` gốc, cùng bash quản lý job.
Một GPU vật lý 0 hoặc 1, một lần chạy với seed=5.

## Setup trên server

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
conda activate magmax
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

Nguồn recipe: `scripts/CIL/all.sh`, `src/args.py`, `finetune_splitted.py`,
`src/utils.py`, `merge_splitted.py`, `src/merging/task_vectors.py` và `src/eval.py`
tại upstream commit `4f49f18fb30216d3a0dbe4220b469a3273e45475`.
Chưa train trên server của bạn; chưa có accuracy hoặc thời gian thực tế của RTX4090.
