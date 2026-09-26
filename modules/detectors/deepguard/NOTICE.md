Vendored from [HanMoonSub/DeepGuard](https://github.com/HanMoonSub/DeepGuard) (MIT, see `LICENSE`):
`deepguard/models/ms_eff_gcvit.py` and `deepguard/layers/*`, unmodified except:

- imports changed from `deepguard.layers.*` to relative `.layers.*`
- `FeatExtractor(pretrained=False)` by default, so building the model does not
  download ImageNet weights (the deepfake checkpoint replaces them anyway)

The pip package `deepguard` is not used because it pins OpenCV 5, ultralytics and
other heavy dependencies that conflict with this project.
Checkpoints: `KoreaPeter/ms-eff-gcvit-deepfake-b0-ff-plus-plus` and
`KoreaPeter/ms-eff-gcvit-deepfake-b0-celeb-df-v2` on Hugging Face (MIT).
