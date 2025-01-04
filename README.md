# 🔥 FireFlow Cog implementation 

> A powerful Python package for semantic image editing using Fast Inversion of Rectified Flow, deployed on Replicate.

## 🎯 Overview

**Core Features**
- 🚀 Fast semantic image editing through flow-based models
- 🎨 Multiple editing strategies
- 🔄 Automatic image resizing and optimization
- 📦 Built-in watermarking and EXIF data handling

## 🛠️ Usage on Replicate

```python
import replicate

output = replicate.run(
    "alexgenovese/fireflow:version",
    input={
        "image": "path/to/image.jpg",
        "source_prompt": "a photo of a red car",
        "target_prompt": "a photo of a blue car",
        "editing_strategy": ["replace_v"],
        "num_steps": 8,
        "inject_step": 1,
        "guidance": 2.0
    }
)
```

## ⚙️ Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| image | Input image to edit | Required | - |
| source_prompt | Description of input image | "" | - |
| target_prompt | Desired output description | Required | - |
| editing_strategy | Editing technique | ["replace_v"] | ["replace_v", "add_q", "add_k"] |
| num_steps | Total timesteps | 8 | 1-30 |
| inject_step | Feature sharing steps | 1 | 1-15 |
| guidance | Guidance scale | 2.0 | 1.0-8.0 |

## 🎨 Editing Strategies

**Available Techniques**
- `replace_v`: Replace value vectors
- `add_q`: Add query vectors
- `add_k`: Add key vectors

## ⚠️ Important Notes

**Image Processing**
- Images larger than 1024x1024 are automatically resized
- Dimensions are adjusted to be divisible by 16
- Output includes embedded watermark and EXIF metadata

**Optimization Tips**
- Increase `num_steps` for higher quality results
- Adjust `guidance` to control editing strength
- Experiment with different `editing_strategy` combinations

## 🔧 Technical Details

**Model Components**
- T5 text encoder
- CLIP vision encoder
- Autoencoder
- Flow-based diffusion model

**Hardware Requirements**
- CUDA-compatible GPU
- Sufficient VRAM for model operations

## 🤝 Contributing
We welcome contributions! If you find this project helpful, please give it a ⭐

---
*Powered by Replicate - Advanced Image Editing Made Simple*