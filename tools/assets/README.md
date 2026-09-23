# DINOv3 ViT-B/16 architecture config

`facebook/dinov3-vitb16-pretrain-lvd1689m` is a gated repo, but TurboVLA does not
need its weights: the vision tower is fine-tuned and ships inside the TurboVLA
checkpoint. Only the architecture config and the image normalization are needed,
and those are public. Both files here are verbatim copies from the ungated
[`onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX`](https://huggingface.co/onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX) mirror.

`image_size` in the config is the mirror's 224; TurboVLA runs the tower at 256,
which changes nothing in the model: DINOv3's RoPE table is computed from the
actual pixel grid on every forward pass, not from the config.
