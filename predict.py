from typing import List
import os
import torch
import numpy as np
from PIL import Image, ExifTags
from cog import BasePredictor, Input, Path
from dataclasses import dataclass
from einops import rearrange

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5)
from huggingface_hub import login

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = 'flux-dev'
        self.offload = False
        
        # Load token from environment variable
        login(token=os.getenv('HUGGINGFACE_TOKEN'))
        
        # Load models
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.torch_device)
        self.t5 = load_t5(self.device, max_length=512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(self.name, device="cpu" if self.offload else self.torch_device)

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image to edit"),
        source_prompt: str = Input(description="Description of the input image", default=""),
        target_prompt: str = Input(description="Description of the desired output image"),
        editing_strategy: List[str] = Input(
            description="Editing technique to use",
            choices=["replace_v", "add_q", "add_k"],
            default=["replace_v"]
        ),
        num_steps: int = Input(
            description="Total number of timesteps",
            default=8,
            ge=1,
            le=30
        ),
        inject_step: int = Input(
            description="Feature sharing steps",
            default=1,
            ge=1,
            le=15
        ),
        guidance: float = Input(
            description="Guidance scale",
            default=2.0,
            ge=1.0,
            le=8.0
        )
    ) -> Path:
        """Run a single prediction on the model"""
        torch.cuda.empty_cache()
        
        # Load and preprocess image
        init_image = Image.open(str(image))
        init_image = init_image.convert('RGB')
        width, height = init_image.size
        
        # Resize if necessary
        if max(width, height) > 1024:
            if height > width:
                new_height = 1024
                new_width = int((new_height / height) * width)
            else:
                new_width = 1024
                new_height = int((new_width / width) * height)
            
            init_image = init_image.resize((new_width, new_height))
            print('[INFO] resize large image to [1024, X].')
        
        # Convert to numpy array
        init_image = np.array(init_image)
        shape = init_image.shape
        
        # Adjust dimensions to be divisible by 16
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:new_h, :new_w, :]
        
        width, height = init_image.shape[0], init_image.shape[1]
        
        # Prepare input tensor
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0).to(self.device)
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)
        
        with torch.no_grad():
            init_image = self.ae.encode(init_image.to()).to(torch.bfloat16)

        # Set up sampling options
        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=None
        )
        
        # Set up info dict for editing
        info = {
            'feature': {},
            'inject_step': min(inject_step, num_steps),
            'reuse_v': False,
            'editing_strategy': " ".join(editing_strategy),
            'start_layer_index': 0,
            'end_layer_index': 37,
            'qkv_ratio': [1.0, 1.0, 1.0]
        }
        
        # Handle model offloading
        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.torch_device), self.clip.to(self.torch_device)
        
        # Prepare inputs
        with torch.no_grad():
            inp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(self.t5, self.clip, init_image, prompt=opts.target_prompt)
        
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.torch_device)
        
        # Inversion and denoising
        with torch.no_grad():
            z, info = denoise(self.model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
            inp_target["img"] = z
            timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))
            x, _ = denoise(self.model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)
        
        # Decode latents
        x = unpack(x.float(), opts.width, opts.height)
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)
        
        # Generate final image
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            x = self.ae.decode(x)
        
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")
        
        # Convert to PIL and save
        output_image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        
        # Add EXIF data
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        exif_data[ExifTags.Base.ImageDescription] = source_prompt
        
        # Save and return path
        output_path = Path("/tmp/output.png")
        output_image.save(str(output_path), exif=exif_data, quality=95, subsampling=0)
        
        return output_path