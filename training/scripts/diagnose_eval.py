"""Diagnose model output magnitudes at multiple frames."""
import sys
import os
import glob

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deni_train.models.temporal_unet import DeniTemporalResidualNet
from deni_train.utils.tonemapping import aces_tonemap
from deni_train.utils.reproject import reproject

_CH_DEPTH = slice(10, 11)
_CH_MOTION = slice(11, 13)
_CH_ALBEDO_D = slice(13, 16)
_CH_ALBEDO_S = slice(16, 19)


def main():
    device = torch.device("cuda")

    ckpt_path = "configs/checkpoints/model_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_config", {})
    base_channels = model_cfg.get("base_channels", 12)
    epoch = ckpt["epoch"]
    val_loss = ckpt["best_val_loss"]
    print(f"base_channels={base_channels}, epoch={epoch}, val_loss={val_loss:.6f}")

    model = DeniTemporalResidualNet(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data_dir = "D:/training_data_temporal_st"
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.safetensors"), recursive=True))
    print(f"Found {len(files)} files")

    # Process first 3 files
    for fi in range(min(3, len(files))):
        t = load_file(files[fi])
        inp_seq = t["input"].float().to(device)
        tgt_seq = t["target"].float().to(device)
        num_frames = inp_seq.shape[0]
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(files[fi])}")
        print(f"Frames: {num_frames}, Shape: {inp_seq.shape[1:]}")

        prev_denoised = None
        prev_depth = None

        with torch.no_grad():
            for t_idx in range(num_frames):
                gbuf = inp_seq[t_idx].unsqueeze(0)
                curr_depth = gbuf[:, _CH_DEPTH]
                motion = gbuf[:, _CH_MOTION]
                B, _, H, W = gbuf.shape

                if prev_denoised is None:
                    repro_d = torch.zeros(1, 3, H, W, device=device)
                    repro_s = torch.zeros(1, 3, H, W, device=device)
                    disoc = torch.zeros(1, 1, H, W, device=device)
                else:
                    repro_d, valid = reproject(prev_denoised[:, :3], motion, prev_depth, curr_depth)
                    repro_s, _ = reproject(prev_denoised[:, 3:6], motion, prev_depth, curr_depth)
                    disoc = valid

                temporal_input = torch.cat([repro_d, repro_s, disoc, gbuf], dim=1)
                pred, raw_wt = model(temporal_input)
                prev_denoised = pred.detach()
                prev_depth = curr_depth.detach()

                target = tgt_seq[t_idx].unsqueeze(0)

                albedo_d = gbuf[:, _CH_ALBEDO_D]
                albedo_s = gbuf[:, _CH_ALBEDO_S]
                pred_rad = pred[:, :3] * albedo_d + pred[:, 3:6] * albedo_s
                tgt_rad = target[:, :3] * albedo_d + target[:, 3:6] * albedo_s
                noisy_rad = gbuf[:, :3] * albedo_d + gbuf[:, 3:6] * albedo_s

                pred_tm = aces_tonemap(pred_rad)
                tgt_tm = aces_tonemap(tgt_rad)
                noisy_tm = aces_tonemap(noisy_rad)

                if t_idx in [0, 7, 15] or t_idx == num_frames - 1:
                    print(f"\n--- Frame {t_idx} ---")
                    print(f"  pred demod:  mean={pred.mean():.4f} std={pred.std():.4f} "
                          f"min={pred.min():.4f} max={pred.max():.4f}")
                    print(f"  tgt  demod:  mean={target.mean():.4f} std={target.std():.4f} "
                          f"min={target.min():.4f} max={target.max():.4f}")
                    print(f"  raw_weight:  mean={raw_wt.mean():.4f} std={raw_wt.std():.4f}")
                    print(f"  pred diff:   mean={pred[:, :3].mean():.4f} "
                          f"spec: mean={pred[:, 3:6].mean():.4f}")
                    print(f"  tgt  diff:   mean={target[:, :3].mean():.4f} "
                          f"spec: mean={target[:, 3:6].mean():.4f}")
                    print(f"  albedo_d:    mean={albedo_d.mean():.4f} "
                          f"albedo_s: mean={albedo_s.mean():.4f}")
                    print(f"  pred rad:    mean={pred_rad.mean():.4f} std={pred_rad.std():.4f}")
                    print(f"  tgt  rad:    mean={tgt_rad.mean():.4f} std={tgt_rad.std():.4f}")
                    print(f"  noisy rad:   mean={noisy_rad.mean():.4f} std={noisy_rad.std():.4f}")
                    print(f"  pred ACES:   mean={pred_tm.mean():.4f} std={pred_tm.std():.4f}")
                    print(f"  tgt  ACES:   mean={tgt_tm.mean():.4f} std={tgt_tm.std():.4f}")
                    print(f"  noisy ACES:  mean={noisy_tm.mean():.4f} std={noisy_tm.std():.4f}")


if __name__ == "__main__":
    main()
