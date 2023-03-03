import argparse
import os
from multiprocessing import Process

import torch.multiprocessing as mp
from diffusers import LDMSuperResolutionPipeline
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str, default="testsets/solafune_test")
parser.add_argument("--output_dir", type=str, default="results/solafune/submit")


def process(args):
    device = "cuda"
    pipe = LDMSuperResolutionPipeline.from_pretrained(
        "CompVis/ldm-super-resolution-4x-openimages"
    )
    pipe = pipe.to(device)

    image_files = os.listdir(args.test_dir)
    for image_file in image_files:
        image_path = os.path.join(args.test_dir, image_file)
        new_image_path = os.path.join(
            args.output_dir, image_file.replace("_low", "_answer")
        )
        low_res_img = Image.open(image_path).convert("RGB")
        low_res_img = low_res_img.resize((128, 128))
        upscaled_image = pipe(low_res_img, num_inference_steps=100, eta=1).images[0]
        upsampled_img = upscaled_image.resize(size=(650, 650), resample=Image.BILINEAR)
        upsampled_img.save(new_image_path)


if __name__ == "__main__":
    args = parser.parse_args()
    process(args)

    """
    mp.set_start_method("spawn")
    process_list = []
    for i in range(2):
        proc = Process(target=process, kwargs={"args": args})
        proc.start()
        process_list.append(proc)

    for proc in process_list:
        proc.join()
    """
