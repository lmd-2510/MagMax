"""MagMax-only evaluation, reusing the upstream task-vector and evaluator code."""
import gc
import json
from pathlib import Path
import subprocess
import time

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.merging.task_vectors import TaskVector, merge_max_abs


def main():
    args = parse_arguments()
    if not args.sequential_finetuning or args.split_strategy != "class":
        raise ValueError("MagMax baseline requires sequential class-incremental fine-tuning.")
    if args.lwf_lamb or args.ewc_lamb:
        raise ValueError("This runner evaluates sequential fine-tuning without LwF/EWC.")
    pretrained = Path(f"checkpoints/{args.model}/zeroshot.pt")
    checkpoint_dir = (Path(f"checkpoints/{args.model}/sequential_finetuning/class_incremental")
                      / f"{args.dataset}-{args.n_splits}"
                      / f"ft-epochs-{args.epochs}-seed:{args.seed}")
    checkpoints = [checkpoint_dir / f"finetuned_{idx}.pt" for idx in range(args.n_splits)]
    for path in [pretrained, *checkpoints]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    output = Path(args.save or f"outs/magmax_{args.dataset}_{args.n_splits}_seed{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    merged = None
    for idx, checkpoint in enumerate(checkpoints):
        print(f"[MERGE] task {idx + 1}/{len(checkpoints)}", flush=True)
        current = TaskVector(str(pretrained), str(checkpoint))
        # Ordered pairwise merging gives the same result and tie behavior as upstream.
        # Only the accumulated vector and one new task vector remain in host RAM.
        merged = current if merged is None else merge_max_abs([merged, current])
        del current
        gc.collect()
    image_encoder = merged.apply_to(str(pretrained), scaling_coef=0.5)
    del merged
    gc.collect()
    # Full 100-class test head + val_preprocess, as in upstream search_evaluate_merging.
    metrics = eval_single_dataset(image_encoder, args.dataset, args)
    merged_path = output / "magmax_merged.pt"
    image_encoder.save(str(merged_path))
    result = {
        "method": "MagMax", "dataset": args.dataset, "model": args.model,
        "n_splits": args.n_splits, "seed": args.seed, "epochs_per_task": args.epochs,
        "merge_coefficient": 0.5, "top1": metrics["top1"],
        "top1_percent": metrics["top1"] * 100,
        "evaluation": "final full-dataset test, full-class head, val_preprocess",
        "merge_and_evaluation_seconds": time.monotonic() - started,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "task_checkpoints": [str(path) for path in checkpoints],
        "pretrained_checkpoint": str(pretrained), "merged_checkpoint": str(merged_path),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[RESULT] MagMax top-1 = {result['top1_percent']:.2f}%", flush=True)
    print(f"[RESULT] {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
