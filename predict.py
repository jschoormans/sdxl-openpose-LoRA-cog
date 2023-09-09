# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import json
from cog import BasePredictor, Input, Path
import os
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, DiffusionPipeline
from safetensors import safe_open
from dataset_and_utils import TokenEmbeddingsHandler

from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.utils import load_image
from safetensors.torch import load_file

CONTROL_NAME = "lllyasviel/ControlNet"
OPENPOSE_NAME = "thibaud/controlnet-openpose-sdxl-1.0"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
CONTROL_CACHE = "control-cache"
POSE_CACHE = "pose-cache"
MODEL_CACHE = "model-cache"
REFINER_CACHE = "refiner-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.openpose = OpenposeDetector.from_pretrained(
            CONTROL_NAME,
            cache_dir=CONTROL_CACHE,
        )
        controlnet = ControlNetModel.from_pretrained(
            POSE_CACHE,
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,   # this is in the sdxl repo
            variant="fp16",   # in sdxl repo
        )
        self.pipe = pipe.to("cuda")
        refiner = DiffusionPipeline.from_pretrained(
            REFINER_CACHE,
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        
        self.refiner = refiner.to("cuda")
        
        # from the sdxl repo 
        print("Loading Unet LoRA")
        self.is_lora=True

        tensors = load_file("lora/lora.safetensors")
        unet = self.pipe.unet
        unet_lora_attn_procs = {}
        name_rank_map = {}
        for tk, tv in tensors.items():
            # up is N, d
            if tk.endswith("up.weight"):
                proc_name = ".".join(tk.split(".")[:-3])
                r = tv.shape[1]
                name_rank_map[proc_name] = r

        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            module = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=name_rank_map[name],
            )
            unet_lora_attn_procs[name] = module.to("cuda")

        self.pipe.unet.set_attn_processor(unet_lora_attn_procs)
        self.pipe.unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [self.pipe.text_encoder, self.pipe.text_encoder_2],
            [self.pipe.tokenizer, self.pipe.tokenizer_2]
        )
        handler.load_embeddings("lora/embeddings.pti")

        # load params
        with open("lora/special_params.json", "r") as f:
            params = json.load(f)
        self.token_map = params
        self.tuned_model = True
        
        

    def predict(
        self,
        image: Path = Input(description="Input pose image"),
        prompt: str = Input(
            description="Input prompt",
            default="a latina ballerina, romantic sunset, 4k photo",
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="low quality, bad quality",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        high_noise_frac: float = Input(
            description="for expert_ensemble_refiner, the fraction of noise to use",
            default=0.8,
            le=1.0,
            ge=0.0,
        ),
        seed: int = Input(description="Random seed. Set to 0 to randomize the seed", default=0),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        
        
    ) -> Path:
        """Run a single prediction on the model"""
        refine = "base_image_refiner" #"expert_ensemble_refiner"
        refine_steps = None

        if (seed is None) or (seed <= 0):
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")

        # Load pose image
        image = Image.open(image).resize((1024, 1024))
        openpose_image = self.openpose(image).resize((1024, 1024))

        sdxl_kwargs = {}
        if refine == "expert_ensemble_refiner":
            sdxl_kwargs["output_type"] = "latent"
            sdxl_kwargs["denoising_end"] = high_noise_frac

        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
                print(f"Prompt: {prompt}")
        else:
            print(f"Prompt: {prompt}")
            
            
        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

            
        common_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        print('Starting pipe with args:')
        print(common_args)
        print(sdxl_kwargs)
        
        output = self.pipe(
            **common_args,
            image=openpose_image,
            **sdxl_kwargs
        )

        if refine in ["expert_ensemble_refiner", "base_image_refiner"]:
            refiner_kwargs = {
                "image": output.images,
            }
            if refine == "expert_ensemble_refiner":
                refiner_kwargs["denoising_start"] = high_noise_frac
            if refine == "base_image_refiner" and refine_steps:
                common_args["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args, **refiner_kwargs)
        
        output_path = "./output.png"
        output_image = output.images[0]
        output_image.save(output_path)

        return Path(output_path)