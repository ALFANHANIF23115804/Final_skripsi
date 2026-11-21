"""Bandingkan fitur tumor (ukuran, lokasi, kontras) dan kualitas citra
antara prediksi benar vs salah, memakai mask segmentasi BRISC2025 (test split).

Jalankan:  python scripts/analyze_error_features.py
Output  :  results/error_feature_analysis/{features.csv, summary.txt}
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "per_class_evaluation" / "predictions.csv"
SEG = ROOT / "brisc2025-v6" / "brisc2025" / "segmentation_task" / "test"
OUT = ROOT / "results" / "error_feature_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def features(img_path: Path, mask_path: Path) -> dict:
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    h, w = img.shape

    head = img > 20                      # kepala vs latar hitam
    head = ndi.binary_fill_holes(ndi.binary_opening(head, iterations=2))
    ys, xs = np.nonzero(head)
    if len(ys) == 0:
        ys, xs = np.array([0, h - 1]), np.array([0, w - 1])
    hy0, hy1, hx0, hx1 = ys.min(), ys.max(), xs.min(), xs.max()

    f = {"img_h": h, "img_w": w,
         "head_area_frac": float(head.mean())}

    # ---- fitur tumor ----
    area = int(mask.sum())
    f["tumor_area_px"] = area
    f["tumor_area_frac_head"] = area / max(int(head.sum()), 1)
    if area > 0:
        ty, tx = np.nonzero(mask)
        cy, cx = ty.mean(), tx.mean()
        # posisi relatif terhadap kotak kepala: 0..1 (x: kiri-kanan, y: atas-bawah)
        f["tumor_cx_rel"] = float((cx - hx0) / max(hx1 - hx0, 1))
        f["tumor_cy_rel"] = float((cy - hy0) / max(hy1 - hy0, 1))
        # jarak dari pusat kepala (dinormalisasi tengah->tepi)
        hcx, hcy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
        f["tumor_dist_center"] = float(np.hypot((cx - hcx) / max((hx1 - hx0) / 2, 1),
                                                (cy - hcy) / max((hy1 - hy0) / 2, 1)))
        # bentuk: solidity-kasar (area / luas bounding box) dan jumlah komponen
        bw, bh = tx.max() - tx.min() + 1, ty.max() - ty.min() + 1
        f["tumor_bbox_fill"] = area / float(bw * bh)
        f["tumor_n_components"] = int(ndi.label(mask)[1])
        # keteraturan tepi: keliling^2 / (4*pi*luas) -> 1 = bulat sempurna
        er = ndi.binary_erosion(mask)
        perim = int((mask & ~er).sum())
        f["tumor_irregularity"] = perim ** 2 / (4 * np.pi * area)
        # kontras: tumor vs cincin sekitar (dilasi - tumor)
        ring = ndi.binary_dilation(mask, iterations=8) & ~ndi.binary_dilation(mask, iterations=2)
        ring &= head
        inside = img[mask]
        f["tumor_mean"] = float(inside.mean())
        f["tumor_std"] = float(inside.std())          # heterogenitas
        if ring.sum() > 10:
            f["tumor_ring_contrast"] = float(inside.mean() - img[ring].mean())
            f["tumor_cnr"] = float((inside.mean() - img[ring].mean()) / (img[ring].std() + 1e-6))
        else:
            f["tumor_ring_contrast"] = f["tumor_cnr"] = float("nan")

    # ---- kualitas citra ----
    f["global_contrast_std"] = float(img[head].std())
    f["mean_intensity_head"] = float(img[head].mean())
    lap = ndi.laplace(img)
    f["sharpness_lapvar"] = float(lap[head].var())
    bg = ~ndi.binary_dilation(head, iterations=10)
    f["bg_noise_std"] = float(img[bg].std()) if bg.sum() > 100 else float("nan")
    f["snr_head_over_bg"] = (f["mean_intensity_head"] / f["bg_noise_std"]
                             if f["bg_noise_std"] and f["bg_noise_std"] > 0 else float("nan"))
    f["pad_black_rows_frac"] = float((img.max(axis=1) < 10).mean())  # bar hitam/padding
    return f


def main() -> int:
    rows = list(csv.DictReader(open(PRED, encoding="utf-8")))
    out_rows, skipped = [], 0
    for r in rows:
        stem = Path(r["image_path"]).stem
        if r["true_label"] == "no_tumor":
            continue
        img_p = SEG / "images" / f"{stem}.jpg"
        m_p = SEG / "masks" / f"{stem}.png"
        if not (img_p.exists() and m_p.exists()):
            skipped += 1
            continue
        parts = stem.split("_")
        d = {"file": stem, "true_label": r["true_label"],
             "pred_label": r["predicted_label"], "plane": parts[-2],
             "correct": r["correct"] == "True",
             "confidence": float(r["confidence"])}
        d.update(features(img_p, m_p))
        out_rows.append(d)

    cols = list(out_rows[0].keys())
    for d in out_rows:
        for c in cols:
            d.setdefault(c, "")
    with open(OUT / "features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    wrong = [d for d in out_rows if not d["correct"]]
    right = [d for d in out_rows if d["correct"]]
    lines = [f"Gambar tumor dianalisis: {len(out_rows)} (skip tanpa mask: {skipped})",
             f"Benar: {len(right)}  Salah: {len(wrong)}", ""]
    feats = [c for c in cols if c not in
             ("file", "true_label", "pred_label", "plane", "correct", "confidence",
              "img_h", "img_w")]
    lines.append(f"{'fitur':26s}{'median_benar':>14s}{'median_salah':>14s}{'p (MWU)':>10s}")
    for c in feats:
        a = np.array([d[c] for d in right], dtype=float)
        b = np.array([d[c] for d in wrong], dtype=float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 5 or len(b) < 3:
            continue
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        lines.append(f"{c:26s}{np.median(a):14.3f}{np.median(b):14.3f}{p:10.3f}")

    # per kelas: ukuran & lokasi median
    lines += ["", "Per kelas (semua gambar tumor): median area_frac, cx, cy, ring_contrast"]
    for cls in ("glioma", "meningioma", "pituitary"):
        sub = [d for d in out_rows if d["true_label"] == cls]
        med = lambda k: np.nanmedian([float(d[k]) for d in sub])
        lines.append(f"{cls:11s} n={len(sub):3d} area={med('tumor_area_frac_head'):.4f} "
                     f"cx={med('tumor_cx_rel'):.2f} cy={med('tumor_cy_rel'):.2f} "
                     f"contrast={med('tumor_ring_contrast'):.1f}")

    # persentil tiap kesalahan terhadap kelas aslinya
    lines += ["", "Kesalahan: persentil fitur terhadap SEMUA gambar kelas aslinya (0=terkecil,100=terbesar)"]
    keys = ["tumor_area_frac_head", "tumor_cx_rel", "tumor_cy_rel",
            "tumor_ring_contrast", "sharpness_lapvar", "global_contrast_std"]
    lines.append(f"{'file':44s}{'asli->pred':24s}" + "".join(f"{k[:11]:>12s}" for k in keys))
    for d in sorted(wrong, key=lambda x: x["confidence"]):
        sub = [float(x[k]) for x in out_rows if x["true_label"] == d["true_label"]
               for k in [keys[0]]]
        pcts = []
        for k in keys:
            ref = np.array([float(x[k]) for x in out_rows if x["true_label"] == d["true_label"]])
            ref = ref[~np.isnan(ref)]
            v = float(d[k])
            pcts.append(100.0 * (ref < v).mean() if not np.isnan(v) else float("nan"))
        lines.append(f"{d['file'][-26:]:44s}{(d['true_label'][:4] + '->' + d['pred_label'][:4]):24s}"
                     + "".join(f"{p:12.0f}" for p in pcts))

    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
