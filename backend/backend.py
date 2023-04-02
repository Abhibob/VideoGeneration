# fastapi setup
from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel
import pymongo
from sys import argv
import os

def display_video(video_path):
    video = imageio.mimread(video_path)
    fig = plt.figure(figsize=(4.2,4.2))  #Display size specification
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    mov = []
    for i in range(len(video)):  #Append videos one by one to mov
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        mov.append([img])

    #Animation creation
    anime = animation.ArtistAnimation(fig, mov, interval=100, repeat_delay=1000)

    plt.close()
    return HTML(anime.to_html5_video())

client = pymongo.MongoClient(argv[1])
videodb = client["videodatabase"]

class Prompts(BaseModel):
    name: str
    strings: List[str]

class Collection(BaseModel):
    name: str

app = FastAPI()

# ml pipeline setup
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from IPython.display import HTML
from base64 import b64encode
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

@app.post("/generate/")
async def create_item(request: Request):
    r = await request.json()
    name = r["name"]
    del r["name"]
    prompts = r.keys()
    collection = videodb[name]
    os.mkdir("videos/" + name)
    for val in prompts:
        collection.insert_one({"prompt": val, "text": r[val], "complete": false})

    for val in prompts:
        video_frames = pipe(r[val], negative_prompt="low quality", num_inference_steps=50, num_frames=50).frames
        video_path = export_to_video(video_frames)
        display_video(video_path)