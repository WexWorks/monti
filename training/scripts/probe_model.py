"""Quick diagnostic: probe model output on real training data."""
import torch
from deni_train.models.temporal_unet import DeniTemporalResidualNet
from deni_train.data.temporal_safetensors_dataset import TemporalSafetensorsDataset
from deni_train.train_temporal import _build_temporal_input
from deni_train.utils.tonemapping import aces_tonemap


def main():
    ckpt = torch.load("configs/checkpoints/model_best.pt", map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in ckpt.get("model_config", {}).items() if k != "type"}
    model = DeniTemporalResidualNet(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Epoch: {ckpt['epoch']}, val_loss: {ckpt.get('best_val_loss'):.4f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Check output conv
    b = model.out_conv.bias.data
    w = model.out_conv.weight.data
    print(f"\nout_conv weight stats: mean={w.mean():.6f}, std={w.std():.6f}")
    print(f"out_conv bias: delta_d={b[:3].tolist()}, delta_s={b[3:6].tolist()}")
    print(f"blend_weight bias: {b[6]:.4f} -> sigmoid={torch.sigmoid(b[6]):.4f}")

    # Load real data
    ds = TemporalSafetensorsDataset("D:/training_data_temporal_st")
    sample = ds[0]
    inp = sample["input"]   # (16, 19, 384, 384)
    tgt = sample["target"]  # (16, 6, 384, 384)

    # Frame 0 (no history)
    gb = inp[0].unsqueeze(0)
    target_f0 = tgt[0].unsqueeze(0)
    temporal_in = _build_temporal_input(gb, None, None)

    with torch.no_grad():
        pred, rw = model(temporal_in)

    print(f"\n=== Frame 0 (no history) ===")
    print(f"Target:    mean={target_f0.mean():.4f}, std={target_f0.std():.4f}, range=[{target_f0.min():.2f}, {target_f0.max():.2f}]")
    print(f"Predicted: mean={pred.mean():.4f}, std={pred.std():.4f}, range=[{pred.min():.2f}, {pred.max():.2f}]")
    print(f"Noisy in:  mean={gb[:, :6].mean():.4f}, std={gb[:, :6].std():.4f}")
    print(f"raw_weight: mean={rw.mean():.4f}, range=[{rw.min():.4f}, {rw.max():.4f}]")

    # Check tonemapped comparison (what evaluation sees)
    albedo_d = gb[:, 13:16]
    albedo_s = gb[:, 16:19]
    pred_rad = pred[:, :3] * albedo_d + pred[:, 3:6] * albedo_s
    tgt_rad = target_f0[:, :3] * albedo_d + target_f0[:, 3:6] * albedo_s
    noisy_rad = gb[:, :3] * albedo_d + gb[:, 3:6] * albedo_s

    pred_rad_tm = aces_tonemap(pred_rad)
    tgt_rad_tm = aces_tonemap(tgt_rad)
    noisy_rad_tm = aces_tonemap(noisy_rad)

    print(f"\n=== Tonemapped radiance ===")
    print(f"Target TM:    mean={tgt_rad_tm.mean():.4f}, range=[{tgt_rad_tm.min():.4f}, {tgt_rad_tm.max():.4f}]")
    print(f"Predicted TM: mean={pred_rad_tm.mean():.4f}, range=[{pred_rad_tm.min():.4f}, {pred_rad_tm.max():.4f}]")
    print(f"Noisy TM:     mean={noisy_rad_tm.mean():.4f}, range=[{noisy_rad_tm.min():.4f}, {noisy_rad_tm.max():.4f}]")

    # Ratio check
    if tgt_rad_tm.mean() > 0:
        print(f"Pred/Target brightness ratio: {pred_rad_tm.mean() / tgt_rad_tm.mean():.4f}")

    # Frame 8 (with history from previous frames)
    prev_denoised = None
    prev_depth = None
    for t in range(9):
        gb_t = inp[t].unsqueeze(0)
        temporal_in_t = _build_temporal_input(gb_t, prev_denoised, prev_depth)
        with torch.no_grad():
            pred_t, rw_t = model(temporal_in_t)
        prev_denoised = pred_t.detach()
        prev_depth = gb_t[:, 10:11].detach()

    target_f8 = tgt[8].unsqueeze(0)
    print(f"\n=== Frame 8 (with history accumulation) ===")
    print(f"Target:    mean={target_f8.mean():.4f}, std={target_f8.std():.4f}")
    print(f"Predicted: mean={pred_t.mean():.4f}, std={pred_t.std():.4f}")
    print(f"raw_weight: mean={rw_t.mean():.4f}")


if __name__ == "__main__":
    main()
