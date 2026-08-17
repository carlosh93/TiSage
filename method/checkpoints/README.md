# Checkpoints

## Included Prior Heads

- `pretrained/medsiglip_head_dfutissue.pt`
- `pretrained/medsiglip_head_lutseg.pt`

These small classifier heads are used by TiSage and the classifier-based
prior-only evaluations.

## Segmentation Checkpoints

Four trained checkpoints associated with the paper settings are public at
[ksanchez84/TiSage](https://huggingface.co/ksanchez84/TiSage). They are not
committed to Git because each file is approximately 866 MB. Expected filenames
and SHA-256 hashes are listed in `segmentation_checkpoints.sha256`.

These `best.pth` files are best-student snapshots and contain the EMA state from
the same epoch. They are provided for inference, inspection, and Figure 4; the
paper's best-EMA Table 1 values remain traceable to the committed evidence logs.

Figure 4 requires:

- `lutseg_unimatch_v2_1_8_seed0_best.pth`
- `lutseg_tisage_1_8_seed0_best.pth`

Download those two files into an ignored local directory:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ksanchez84/TiSage', local_dir='method/checkpoints/downloaded', allow_patterns=['lutseg_*.pth'])"
```

Verify downloaded files from the repository root with:

```bash
cd method/checkpoints/downloaded
sha256sum --check ../segmentation_checkpoints.sha256 --ignore-missing
```

The model card, checkpoint scope, and license notice are mirrored in
`HF_MODEL_CARD.md` and `NOTICE.md`.
