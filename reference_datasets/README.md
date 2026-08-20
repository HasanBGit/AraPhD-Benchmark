# Reference datasets — how the originals structure their Q&A

Downloaded samples of the datasets ArabPhD's modes are directly modeled on,
so we can see their actual question/answer structure before building our
own equivalents. Not full copies — sample records + a handful of example
images per source, enough to see the format. Full re-download instructions
are in each subfolder's README.

| Folder | Covers ArabPhD mode(s) | Source |
|---|---|---|
| `phd_original_liu2025/` | Mode 1–4 (base, sec≈iac, icc, ccs) | Liu et al., CVPR 2025 — [PhD](https://github.com/jiazhen-code/PhD) |
| `mm_upd_miyai2025/` | Mode 5 (nota) | Miyai et al., ACL 2025 — [MM-UPD](https://huggingface.co/datasets/MM-UPD/MM-UPD) |

See `mm_upd_miyai2025/README.md` for an important note: MM-UPD's "3 things"
(AAD/IASD/IVQD) are three *types of unsolvable question*, not the same as
the MCDR/OEDR/UDR three *prompting conditions* our own nota design doc
describes (those come from a different, newer paper — Wang et al. 2026 —
which doesn't have a public dataset to download yet).
