import os
#try:
#    import torchaudio
#except ImportError:
#    os.system("cd ./F5-TTS; pip install -e .")
#os.system("pip install -U gradio")


import spaces
import logging
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
import torchaudio

import tempfile

import requests
import shutil
import numpy as np

from huggingface_hub import hf_hub_download

if True:
    model_path = "./MMAudio/weights/"
    
    file_path = hf_hub_download(repo_id="lshzhm/DeepAudio-V1", filename="MMAudio/mmaudio_small_44k.pth", local_dir=model_path)
    print(f"Model saved at: {file_path}")
    shutil.move("./MMAudio/weights/MMAudio/mmaudio_small_44k.pth", "./MMAudio/weights/")
    
    model_path = "./MMAudio/ext_weights/"
    
    file_path = hf_hub_download(repo_id="lshzhm/DeepAudio-V1", filename="MMAudio/v1-44.pth", local_dir=model_path)
    print(f"Model saved at: {file_path}")
    shutil.move("./MMAudio/ext_weights/MMAudio/v1-44.pth", "./MMAudio/ext_weights/")
    file_path = hf_hub_download(repo_id="lshzhm/DeepAudio-V1", filename="MMAudio/synchformer_state_dict.pth", local_dir=model_path)
    print(f"Model saved at: {file_path}")
    shutil.move("./MMAudio/ext_weights/MMAudio/synchformer_state_dict.pth", "./MMAudio/ext_weights/")
    
    
    model_path = "./F5-TTS/ckpts/v2c/"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_path = hf_hub_download(repo_id="lshzhm/DeepAudio-V1", filename="v2c_s44.pt", local_dir=model_path)

    print(f"Model saved at: {file_path}")


log = logging.getLogger()


import sys
sys.path.insert(0, "./MMAudio/")
from demo import v2a_load, v2a_infer

v2a_loaded = v2a_load()


import sys
sys.path.insert(0, "./F5-TTS/src/")
from f5_tts.infer.infer_cli_test import v2s_infer


#@spaces.GPU(duration=120)
def video_to_audio_and_speech(video: gr.Video, prompt: str, v2a_num_steps: int, text: str, audio_prompt: gr.Audio, text_prompt: str, v2s_num_steps: int):

    video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    audio_p_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    
    output_dir = os.path.dirname(video_path)
    video_save_path = str(output_dir) + "/" + str(video_path).replace("/", "__").strip(".") + ".mp4"
    
    print("paths", video, video_path, output_dir, video_save_path)
    print("paths", audio_prompt, audio_p_path, audio_prompt[1].shape, audio_prompt[1].max(), audio_prompt[1].min(), type(audio_prompt[1]))

    if video.startswith("http"):
        data = requests.get(video, timeout=60).content
        with open(video_path, "wb") as fw:
            fw.write(data)
    else:
        shutil.copy(video, video_path)
    
    if isinstance(audio_prompt, tuple):
        sr, data = audio_prompt
        torchaudio.save(audio_p_path, torch.from_numpy(data.reshape(1,-1)/32768.0).to(torch.float32), sr)
    elif audio_prompt.startswith("http"):
        data = requests.get(audio_prompt, timeout=60).content
        with open(audio_p_path, "wb") as fw:
            fw.write(data)
    else:
        shutil.copy(audio_prompt, audio_p_path)
    
    #if prompt == "":
    #    command = "cd ./MMAudio; python ./demo.py --variant small_44k --output %s --video %s --calc_energy 1 --num_steps %d" % (output_dir, video_path, v2a_num_steps)
    #else:
    #    command = "cd ./MMAudio; python ./demo.py --variant small_44k --output %s --video %s --prompt %s --calc_energy 1 --num_steps %d" % (output_dir, video_path, prompt, v2a_num_steps)
    #print("v2a command", command)
    #os.system(command)
    
    
    v2a_infer(output_dir, video_path, prompt, v2a_num_steps, v2a_loaded)
    
    
    video_gen = video_save_path[:-4]+".mp4.gen.mp4"
    
    #command = "python ./F5-TTS/src/f5_tts/infer/infer_cli_test.py --output_dir %s --start 0 --end 1 --ckpt_file ./F5-TTS/ckpts/v2c/v2c_s44.pt --v2a_path %s --wav_p %s --txt_p \"%s\" --video %s --v2a_wav %s --txt \"%s\" --nfe_step %d" % (output_dir, output_dir, audio_p_path, text_prompt, video_save_path, video_save_path[:-4]+".flac", text, v2s_num_steps)
    #print("v2s command", command, video_gen)
    #os.system(command)
    
    
    v2s_infer(output_dir, output_dir, audio_p_path, text_prompt, video_save_path, video_save_path[:-4]+".flac", text, v2s_num_steps)
    
    
    return video_save_path, video_gen


video_to_audio_and_speech_tab = gr.Interface(
    fn=video_to_audio_and_speech,
    description="""
    Project page: <a href="https://acappemin.github.io/DeepAudio-V1.github.io">https://acappemin.github.io/DeepAudio-V1.github.io</a><br>
    Code: <a href="https://github.com/acappemin/DeepAudio-V1">https://github.com/acappemin/DeepAudio-V1</a><br>
    """,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Text(label='Video-to-Audio Text Prompt'),
        gr.Number(label='Video-to-Audio Num Steps', value=25, precision=0, minimum=1),
        gr.Text(label='Video-to-Speech Transcription'),
        gr.Audio(label='Video-to-Speech Speech Prompt'),
        gr.Text(label='Video-to-Speech Speech Prompt Transcription'),
        gr.Number(label='Video-to-Speech Num Steps', value=32, precision=0, minimum=1),
    ],
    outputs=[
        gr.Video(label="Video-to-Audio Output"),
        gr.Video(label="Video-to-Speech Output"),
    ],
    cache_examples=False,
    title='Video-to-Audio-and-Speech',
    examples=[
        [
            './tests/0235.mp4',
            '',
            25,
            "Who finally decided to show up for work Yay",
            './tests/Gobber-00-0778.wav',
            "I've still got a few knocking around in here",
            32,
        ],
        [
            './tests/0778.mp4',
            '',
            25,
            "I've still got a few knocking around in here",
            './tests/Gobber-00-0235.wav',
            "Who finally decided to show up for work Yay",
            32,
        ],
    ])


if __name__ == "__main__":
    gr.TabbedInterface([video_to_audio_and_speech_tab], ['Video-to-Audio-and-Speech']).queue(max_size=1).launch(server_name='0.0.0.0', server_port=30458)

