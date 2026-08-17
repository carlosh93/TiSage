# Checkpoint Release Notice

The four segmentation checkpoint files in `ksanchez84/TiSage` are distributed
under the Apache License 2.0. They include DINOv2-derived backbone parameters;
the upstream DINOv2 implementation and weights are provided by Meta under
Apache License 2.0. The full license text is included as `LICENSE` in the model
repository and as `licenses/Apache-2.0.txt` in the code repository.

TiSage-specific source code is separately distributed under the MIT License in
the code repository. The LUTSeg dataset is distributed under CC BY 4.0, and
DFUTissue retains its source dataset terms. Dataset licenses do not change by
being referenced from the model repository.

Training used `google/medsiglip-448` as a frozen semantic-prior model. MedSigLIP
parameters are not contained in the released segmentation checkpoints. Users
who retrain TiSage must obtain MedSigLIP directly from its publisher and accept
the applicable Health AI Developer Foundations terms.
