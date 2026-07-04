# Pretrained Checkpoints

Three trained DINOv3 + HRDecoder checkpoints ship with the submission.

| File | Training run | Use case |
|------|--------------|----------|
| `dinov3_hrdecoder_full_best_loss=0.0615.ckpt` | `train_full.py`, no val | **Submission default** — full-data train, fixed 50-epoch budget |
| `dinov3_hrdecoder_val_best_mIoU=0.5500_epoch40.ckpt` | `train.py`, 15 % val | Best validation mIoU |
| `dinov3_hrdecoder_best_acc_miou=0.5153_acc=0.9562.ckpt` | `train.py`, 15 % val | Best overall accuracy |

Each `.ckpt` is ~2.3 GB.

`pretrained/` holds **shipped** models — read-only artefacts that came
with the bundle. New training runs (`train.py` / `train_full.py`)
write their own checkpoints to **`../checkpoints/`** instead.

## Default inference checkpoint

`run_pipeline.py --checkpoint` defaults to
`pretrained/dinov3_hrdecoder_full_best_loss=0.0615.ckpt`. To use a
different one:

```bash
python -m dinov3_hrdecoder_pipeline.inference.run_pipeline \
    --checkpoint pretrained/dinov3_hrdecoder_val_best_mIoU=0.5500_epoch40.ckpt
```

## Symlinks (current state)

To keep the bundle small during development on a slow remote mount,
the three files here are **symlinks** pointing back to the original
training-output location. Verify they resolve:

```bash
for f in pretrained/*.ckpt; do
  echo "$f → $(readlink "$f")"
  [ -f "$(readlink "$f")" ] && echo "  OK ($(ls -lh "$(readlink "$f")" | awk '{print $5}'))"
done
```

## Replacing symlinks with real files (before shipping)

Before zipping the bundle for submission, replace each symlink with
the actual file:

```bash
cd pretrained
for f in *.ckpt; do
  if [ -L "$f" ]; then
    target=$(readlink "$f")
    rm "$f"
    cp "$target" "$f"
  fi
done
```

If the source filesystem is a slow remote mount, **run this step on
the remote machine** (where it becomes a local `cp`) and fetch the
result with `rsync -avz`.
