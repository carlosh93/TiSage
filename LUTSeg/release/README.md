# Hugging Face Release Builder

Build the public LUTSeg dataset in a new staging directory without modifying
the source dataset:

```bash
python LUTSeg/release/build_hf_release.py \
  --source /path/to/LUTSeg \
  --output /tmp/lutseg-hf-release
```

The builder removes embedded JPEG metadata without changing decoded pixels,
rewrites masks as metadata-free PNGs, selects the 46-case multi-expert gold
subset, copies the paper splits, generates checksums, and runs the validator.

Validate an existing staging directory independently:

```bash
python LUTSeg/release/validate_hf_release.py /tmp/lutseg-hf-release
```

The same command can validate a directory downloaded with
`huggingface_hub.snapshot_download`; Hugging Face's generated `.gitattributes`
and local `.cache/` metadata are excluded from the checksum-manifest comparison.

The staging directory is the only directory that should be uploaded. Never
upload the original dataset tree or `.env`.
