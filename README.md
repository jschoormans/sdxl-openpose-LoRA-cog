# openpose-controlnet SDXL with custom LoRa

This is a cog implementation of SDXL with LoRa, trained with Replicate's [Fine-tune SDXL with your own images
](https://replicate.com/blog/fine-tune-sdxl).

This is based on [thibaud/controlnet-openpose-sdxl-1.0](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0) and [lucataco/cog-sdxl-controlnet-openpose](https://github.com/lucataco/cog-sdxl-controlnet-openpose)


## Example:

Inputs:

"Person wearing a TOK shirt"

![alt text](demo2.png)

Output

![alt text](example.png)

with the custom LoRA SDXL model [jschoormans/zara-striped-shirt](https://replicate.com/jschoormans/zara-striped-shirt/versions/28ec2be75b583952776c0947fac0cc3985b87c0c2b165f42711f03972ac3771a)


## Install and run

First, train your custom model (as a LoRa) on [Replicate](https://replicate.com/blog/fine-tune-sdxl). Then when you are happy with the results, clone this repo and modify,

    git clone https://github.com/jschoormans/cog-sdxl-controlnet-openpose/
    cd cog-sdxl-controlnet-openpose/

If required, install cog:

    sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
    sudo chmod +x /usr/local/bin/cog


Download the weights: SDXL, controlnet weights, and your LoRA. Change the URL in the script to your own weights URL before running.

    cog run script/download-weights 


Then, you can run predictions. Use the paramater lora_scale to control LoRA strength: 

    cog predict -i image=@demo.jpg -i prompt="a person wearing a TOK, 4k photo" -i lora_scale=0.8


