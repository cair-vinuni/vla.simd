# DINOv3 ViT-B/16 architecture config

TurboVLA's checkpoint contains its fine-tuned DINOv3 vision weights. Conversion
needs the architecture configuration and image normalization settings from the
gated `facebook/dinov3-vitb16-pretrain-lvd1689m` model, but does not download its
weights. The two JSON files here are unchanged copies from the public
[ONNX Community mirror][mirror], revision `d704d636f7b114347fd2a9d6fecac5e1ef464db3`.

The configuration records an `image_size` of 224; TurboVLA uses 256×256 input.
That changes the patch grid, while the learned weights retain the same shapes.
DINOv3 computes its RoPE positions from the input grid on each forward pass.
The converter uses the bundled image mean and standard deviation.

Use the [TurboVLA environment](../../README.md#converter-environments) when
converting. The Transformers cap preserves the hidden-state normalization that
the trained policy expects.

[mirror]: https://huggingface.co/onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX/tree/d704d636f7b114347fd2a9d6fecac5e1ef464db3
