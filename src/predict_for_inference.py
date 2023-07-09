import os
import torch
import wavfile
import python_speech_features
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, CenterCrop
from models.fakenet import FakeNet
from dataset.transforms import ToTensorVideo, NormalizeVideo


def pil_loader(img_path, grayscale=False):
    if grayscale:
        return Image.open(img_path).convert('L')
    else:
        return Image.open(img_path).convert('RGB')

def predict(data_root, model_path, device_id):
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
    FRAMES_PER_CLIPS = 5
    FPS = 25
    model = FakeNet(
        backbone=['c3d_resnet18', 'seresnet18', 'transformer'],
        last_dim=512,
        frames_per_clip=FRAMES_PER_CLIPS,
        img_in_dim=1,
        mode='VA',
        predict_label=True,
        aud_feat='mfcc',
    )
    checkpoint = torch.load(model_path, map_location=device)
    if 'module' in list(checkpoint['model'].keys())[0]:
        model = torch.nn.DataParallel(model, device_ids=[device_id]).to(device)
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
        model = torch.nn.DataParallel(model, device_ids=[device_id]).to(device)
    vid_transform = Compose(
                        [ToTensorVideo(),
                            CenterCrop((224, 224)),
                            NormalizeVideo((0.421,), (0.165,)),]
                    )
    sig = nn.Sigmoid()
    all_logits, all_probs = {}, {}
    for video_name in sorted(os.listdir(data_root)):
        face_img_dir = os.path.join(data_root, video_name, 'faces')
        imgs = []
        for img_name in sorted(os.listdir(face_img_dir)):
            img = np.array(pil_loader(os.path.join(face_img_dir, img_name), grayscale=True))
            imgs.append(img)
        imgs = np.array(imgs)
        audio, sr, _ = wavfile.read(os.path.join(data_root, video_name, 'audio.wav'))
        audio = np.array(audio)

        logits = []
        probs = []

        for i in range(0, imgs.shape[0] - imgs.shape[0]%FRAMES_PER_CLIPS, FRAMES_PER_CLIPS):
            vid = imgs[i:i+FRAMES_PER_CLIPS]
            vid = torch.from_numpy(vid).unsqueeze(-1)
            vid = vid_transform(vid).unsqueeze(0)

            aud = audio[int(i*sr/FPS):int((i+FRAMES_PER_CLIPS)*sr/FPS)]
            aud = python_speech_features.mfcc(aud, samplerate=sr)
            aud = torch.from_numpy(aud).unsqueeze(0).unsqueeze(0)

            vid = vid.to(device, dtype=torch.float)
            aud = aud.to(device, dtype=torch.float)
            output = model((vid, aud))
            logit = output['logits'].detach().cpu().item()
            prob = sig(output['logits']).detach().cpu().item()
            logits.append(logit)
            probs.append(prob)

        all_logits[video_name] = logits
        all_probs[video_name] = probs
    return all_probs