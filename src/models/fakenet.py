from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.c3d_resnet18 import C3dResnet18
from models.vgg_transformer import VGGTransformer
from models.vivit import ViViT
from models.c3dr50 import C3DR50
from models.resnet34 import ResNet34
from models.audio_encoder import VGG, VGG2, ResNet, SEResnet
from models.model_utils.tcn import MultiscaleMultibranchTCN
from models.temporal_transformer import TemporalTransformer


class FakeNet(nn.Module):
    def __init__(self, backbone, img_in_dim, last_dim, frames_per_clip, num_classes=1, fake_classes=1, mode='VA', relu_type = 'prelu', predict_label=False, aud_feat='mfcc'):
        super(FakeNet, self).__init__()

        self.backbone = backbone
        self.img_in_dim = img_in_dim
        self.last_dim = last_dim
        self.frames_per_clip = frames_per_clip
        self.mode = mode
        self.aud_feat = aud_feat
        self.predict_label = predict_label

        # video
        if 'V' in self.mode:
            if self.backbone[0] == 'c3d_resnet18':
                self.v_encoder = C3dResnet18(in_dim=img_in_dim, last_dim=self.last_dim, relu_type=relu_type)
            elif self.backbone[0] == 'vgg_transformer':
                self.v_encoder = VGGTransformer(
                                    in_dim=img_in_dim,
                                    frames_per_clip=self.frames_per_clip,
                                    last_dim=self.last_dim,
                                    dropout = 0.,
                                    emb_dropout = 0.,)
            elif self.backbone[0] == 'vivit':
                self.v_encoder = ViViT(
                                    image_size=224,
                                    patch_size=16,
                                    num_frames=self.frames_per_clip,
                                    in_channels=img_in_dim,
                                    dim=self.last_dim,
                                    depth=3,
                                    heads=12)
            elif self.backbone[0] == 'c3dr50':
                self.v_encoder = C3DR50(
                                    in_channels=img_in_dim,
                                    frames_per_clip=self.frames_per_clip,
                                    block_inplanes=[self.last_dim//16, self.last_dim//8, self.last_dim//4, self.last_dim//2]
                                    )
            elif self.backbone[0] == 'resnet34':
                self.v_encoder = ResNet34(
                                    in_channels=img_in_dim,
                                    num_filters=[self.last_dim//8, self.last_dim//4, self.last_dim//2, self.last_dim])
            else:
                raise NotImplementedError
        # audio
        if 'A' in self.mode:
            if self.aud_feat == 'mfcc':
                if self.backbone[1] == 'vgg':
                    self.a_encoder = VGG(
                                        last_dim=self.last_dim,
                                        temporal_half=True,  # for c3dr50
                                    )
                elif self.backbone[1] == 'seresnet18':
                    self.a_encoder = SEResnet(
                                            layers=[2, 2, 2, 2],
                                            num_filters=[self.last_dim//8, self.last_dim//4, self.last_dim//2, self.last_dim],
                                        )
                else:
                    raise NotImplementedError
            elif self.aud_feat == 'melspectrogram':
                if self.backbone[1] == 'vgg':
                    self.a_encoder = VGG2(last_dim=self.last_dim)
                elif self.backbone[1] == 'resnet18':
                    self.a_encoder = ResNet(frames_per_clip=self.frames_per_clip)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError(f'{self.aud_feat}')

        if self.predict_label:

            dim = self.last_dim*2

            if self.backbone[2] == 'transformer':
                self.temporal_classifier = TemporalTransformer(
                    frames_per_clip=self.frames_per_clip,
                    num_classes=num_classes,
                    dim=dim,
                    depth=6,
                    heads=8,
                    mlp_dim=2048,
                    dropout=0.1,
                    emb_dropout=0.1,
                )
            elif self.backbone[2] == 'tcn':
                tcn_options = {
                    "num_layers": 4,
                    "kernel_size": [3, 5, 7],
                    "dropout": 0.2,
                    "dwpw": False,
                    "width_mult": 1,
                }
                hidden_dim = 256
                self.temporal_classifier = MultiscaleMultibranchTCN(
                    input_size=dim,
                    num_channels=[
                        hidden_dim * len(tcn_options["kernel_size"]) * tcn_options["width_mult"]]
                    * tcn_options["num_layers"],
                    num_classes=num_classes,
                    tcn_options=tcn_options,
                    dropout=tcn_options["dropout"],
                    relu_type=relu_type,
                    dwpw=tcn_options["dwpw"],
                )
            elif self.backbone[2] == 'mlp':
                self.temporal_classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(dim, dim),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                    nn.Linear(dim, dim),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                    nn.Linear(dim, num_classes)
                )
            else:
                raise NotImplementedError

        # Loss
        self.CELoss = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    def forward(self, x, y=None, out_feat=False):
        output = {}

        if self.mode == 'V':
            vid, = x
            v_feats = self.forward_vid(vid)
            if out_feat:
                output['vid'] = v_feats
            if self.normalized == '2norm':
                b, t, d = v_feats.shape
                v_feats = rearrange(v_feats, 'b t d -> b (t d)')
                v_feats = F.normalize(v_feats, p=2, dim=-1)
                v_feats = rearrange(v_feats, 'b (t d) -> b t d', d=d)
            logits, cls_feature = self.forward_classification(v_feats)
            output['logits'] = logits
            if out_feat:
                output['cls'] = cls_feature
        elif self.mode == 'VA':
            vid, aud = x
            v_feats = self.forward_vid(vid)
            a_feats = self.forward_aud(aud)

            if out_feat:
                output['vid'] = v_feats
                output['aud'] = a_feats

            if self.predict_label:
                feats = torch.cat((v_feats, a_feats), -1)
                logits, cls_feature = self.forward_classification(feats)
                output['logits'] = logits
                if out_feat:
                    output['cls'] = cls_feature
        else:
            raise NotImplementedError

        if y is not None:
            y = y.squeeze(1) if logits.shape[-1] != y.shape[-1] else y.float()
            bce_loss = self.CELoss(logits, y)
            output['BCE'] = torch.unsqueeze(bce_loss, 0)

        return output

    def forward_vid(self, vid):
        return self.v_encoder(vid)

    def forward_aud(self, aud):
        return self.a_encoder(aud)

    def forward_classification(self, x):
        return self.temporal_classifier(x)
