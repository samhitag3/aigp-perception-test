## convert_isaac_to_gate_contract.py
```bash
uv run --with numpy --with pillow \
  python utils/convert_isaac_to_gate_contract.py \
  data/raw_isaac_0908 \
  --output data/refined_isaac_0908 \
  --overwrite
```

## display_gates_in_mask.ipynb
replace mask path & run notebook

## images_to_video.py
```bash
uv add opencv-python
```

```bash
uv run python utils/images_to_video.py \
  data/refined_target/sim0721-10/images \
  --output data/refined_target/sim0721-10/video.mp4 \
  --fps 60
```

## video_to_images.py
```bash
uv run python utils/video_to_images.py \
  path/to/video.mp4 \
  path/to/output_images
```

Optional filetype specification (default PNG) and filename prefix (default image_):
```bash
uv run python utils/video_to_images.py \
  data/run_001.mp4 \
  data/run_001/images \
  --file-type png \
  --prefix image
```