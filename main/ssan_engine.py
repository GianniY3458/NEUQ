import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import re
from PIL import Image
from torchvision import transforms
from torchvision import models

class Opt:
    def __init__(self,
                 vocab_size: int = 5000,
                 feature_length: int = 1024,
                 part: int = 6,
                 caption_length_max: int = 100):
        self.vocab_size = vocab_size
        self.feature_length = feature_length
        self.part = part
        self.caption_length_max = caption_length_max

def l2norm(x, dim=1, eps=1e-8):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

class TextExtract(nn.Module):

    def __init__(self, opt):
        super(TextExtract, self).__init__()

        self.embedding_local = nn.Embedding(opt.vocab_size, 512, padding_idx=0)
        self.embedding_global = nn.Embedding(opt.vocab_size, 512, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(512, 2048, num_layers=1, bidirectional=True, bias=False)

    def forward(self, caption_id, text_length):

        text_embedding_global = self.embedding_global(caption_id)
        text_embedding_global = self.dropout(text_embedding_global)
        text_embedding_global = self.calculate_different_length_lstm(text_embedding_global, text_length, self.lstm)

        text_embedding_local = self.embedding_local(caption_id)
        text_embedding_local = self.dropout(text_embedding_local)
        text_embedding_local = self.calculate_different_length_lstm(text_embedding_local, text_length, self.lstm)

        return text_embedding_global, text_embedding_local

    def calculate_different_length_lstm(self, text_embedding, text_length, lstm):
        text_length = text_length.view(-1)
        _, sort_index = torch.sort(text_length, dim=0, descending=True)
        _, unsort_index = sort_index.sort()

        sortlength_text_embedding = text_embedding[sort_index, :]
        sort_text_length = text_length[sort_index]
        # print(sort_text_length)
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(
            sortlength_text_embedding,
            sort_text_length.cpu(),  # 修复 CUDA 报错
            batch_first=True)

        # 保证LSTM权重连续，消除警告
        lstm.flatten_parameters()
        packed_feature, _ = lstm(packed_text_embedding)  # [hn, cn]
        total_length = text_embedding.size(1)
        sort_feature = nn.utils.rnn.pad_packed_sequence(packed_feature,
                                                        batch_first=True,
                                                        total_length=total_length)  # including[feature, length]

        unsort_feature = sort_feature[0][unsort_index, :]
        unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
                          + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2

        return unsort_feature.permute(0, 2, 1).contiguous().unsqueeze(3)

class conv(nn.Module):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.LeakyReLU(0.25, inplace=True)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x

class NonLocalNet(nn.Module):
    def __init__(self, opt, dim_cut=8):
        super(NonLocalNet, self).__init__()
        self.opt = opt

        up_dim_conv = []
        part_sim_conv = []
        cur_sim_conv = []
        conv_local_att = []
        for i in range(opt.part):
            up_dim_conv.append(conv(opt.feature_length//dim_cut, 1024, relu=True, BN=True))
            part_sim_conv.append(conv(opt.feature_length, opt.feature_length // dim_cut, relu=True, BN=False))
            cur_sim_conv.append(conv(opt.feature_length, opt.feature_length // dim_cut, relu=True, BN=False))
            conv_local_att.append(conv(opt.feature_length, 512))

        self.up_dim_conv = nn.Sequential(*up_dim_conv)
        self.part_sim_conv = nn.Sequential(*part_sim_conv)
        self.cur_sim_conv = nn.Sequential(*cur_sim_conv)
        self.conv_local_att = nn.Sequential(*conv_local_att)

        self.register_buffer("zero_eye", (torch.eye(opt.part) * -1e6).unsqueeze(0))

        self.lambda_softmax = 1

    def forward(self, embedding):
        embedding = embedding.unsqueeze(3)
        embedding_part_sim = []
        embedding_cur_sim = []

        for i in range(self.opt.part):
            embedding_i = embedding[:, :, i, :].unsqueeze(2)

            embedding_part_sim_i = self.part_sim_conv[i](embedding_i).unsqueeze(2)
            embedding_part_sim.append(embedding_part_sim_i)

            embedding_cur_sim_i = self.cur_sim_conv[i](embedding_i).unsqueeze(2)
            embedding_cur_sim.append(embedding_cur_sim_i)

        embedding_part_sim = torch.cat(embedding_part_sim, dim=2)
        embedding_cur_sim = torch.cat(embedding_cur_sim, dim=2)

        embedding_part_sim_norm = l2norm(embedding_part_sim, dim=1)  # N*D*n
        embedding_cur_sim_norm = l2norm(embedding_cur_sim, dim=1)  # N*D*n
        self_att = torch.bmm(embedding_part_sim_norm.transpose(1, 2), embedding_cur_sim_norm)  # N*n*n
        zero_eye = self.zero_eye.to(self_att.device)
        self_att = self_att + zero_eye.expand(self_att.size(0), -1, -1)
        self_att = F.softmax(self_att * self.lambda_softmax, dim=1)  # .transpose(1, 2).contiguous()
        embedding_att = torch.bmm(embedding_part_sim_norm, self_att).unsqueeze(3)

        embedding_att_up_dim = []
        for i in range(self.opt.part):
            embedding_att_up_dim_i = embedding_att[:, :, i, :].unsqueeze(2)
            embedding_att_up_dim_i = self.up_dim_conv[i](embedding_att_up_dim_i).unsqueeze(2)
            embedding_att_up_dim.append(embedding_att_up_dim_i)
        embedding_att_up_dim = torch.cat(embedding_att_up_dim, dim=2).unsqueeze(3)

        embedding_att = embedding + embedding_att_up_dim

        embedding_local_att = []
        for i in range(self.opt.part):
            embedding_att_i = embedding_att[:, :, i, :].unsqueeze(2)
            embedding_att_i = self.conv_local_att[i](embedding_att_i).unsqueeze(2)
            embedding_local_att.append(embedding_att_i)

        embedding_local_att = torch.cat(embedding_local_att, 2)

        return embedding_local_att.squeeze()

class TextImgPersonReidNet(nn.Module):

    def __init__(self, opt):
        super(TextImgPersonReidNet, self).__init__()

        self.opt = opt
        resnet50 = models.resnet50(pretrained=True)
        self.ImageExtract = nn.Sequential(*(list(resnet50.children())[:-2]))
        self.TextExtract = TextExtract(opt)

        self.global_avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.local_avgpool = nn.AdaptiveMaxPool2d((opt.part, 1))

        conv_local = []
        for i in range(opt.part):
            conv_local.append(conv(2048, opt.feature_length))
        self.conv_local = nn.Sequential(*conv_local)

        self.conv_global = conv(2048, opt.feature_length)

        self.non_local_net = NonLocalNet(opt, dim_cut=2)
        self.leaky_relu = nn.LeakyReLU(0.25, inplace=True)

        self.conv_word_classifier = nn.Sequential(
            nn.Conv2d(2048, 6, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, caption_id, text_length):

        img_global, img_local, img_non_local = self.img_embedding(image)
        txt_global, txt_local, txt_non_local = self.txt_embedding(caption_id, text_length)

        return img_global, img_local, img_non_local, txt_global, txt_local, txt_non_local

    def img_embedding(self, image):

        image_feature = self.ImageExtract(image)

        image_feature_global = self.global_avgpool(image_feature)
        image_global = self.conv_global(image_feature_global).unsqueeze(2)

        image_feature_local = self.local_avgpool(image_feature)
        image_local = []
        for i in range(self.opt.part):
            image_feature_local_i = image_feature_local[:, :, i, :]
            image_feature_local_i = image_feature_local_i.unsqueeze(2)
            image_embedding_local_i = self.conv_local[i](image_feature_local_i).unsqueeze(2)
            image_local.append(image_embedding_local_i)

        image_local = torch.cat(image_local, 2)

        image_non_local = self.leaky_relu(image_local)
        image_non_local = self.non_local_net(image_non_local)

        return image_global, image_local, image_non_local

    def txt_embedding(self, caption_id, text_length):

        text_feature_g, text_feature_l = self.TextExtract(caption_id, text_length)

        text_global, _ = torch.max(text_feature_g, dim=2, keepdim=True)
        text_global = self.conv_global(text_global).unsqueeze(2)

        text_feature_local = []
        for text_i in range(text_feature_l.size(0)):
            text_feature_local_i = text_feature_l[text_i, :, :text_length[text_i]].unsqueeze(0)

            word_classifier_score_i = self.conv_word_classifier(text_feature_local_i)

            word_classifier_score_i = word_classifier_score_i.permute(0, 3, 2, 1).contiguous()
            text_feature_local_i = text_feature_local_i.repeat(1, 1, 1, 6).contiguous()

            text_feature_local_i = text_feature_local_i * word_classifier_score_i

            text_feature_local_i, _ = torch.max(text_feature_local_i, dim=2)

            text_feature_local.append(text_feature_local_i)

        text_feature_local = torch.cat(text_feature_local, dim=0)

        text_local = []
        for p in range(self.opt.part):
            text_feature_local_conv_p = text_feature_local[:, :, p].unsqueeze(2).unsqueeze(2)
            text_feature_local_conv_p = self.conv_local[p](text_feature_local_conv_p).unsqueeze(2)
            text_local.append(text_feature_local_conv_p)
        text_local = torch.cat(text_local, dim=2)

        text_non_local = self.leaky_relu(text_local)
        text_non_local = self.non_local_net(text_non_local)

        return text_global, text_local, text_non_local

# process_ICFG_data.py / process_CUHK_data.py
class Word2Index(object):
    def __init__(self, vocab):
        self._vocab = {w: index + 1 for index, w in enumerate(vocab)}
        self.unk_id = len(vocab) + 1
    def __call__(self, word):
        if word not in self._vocab:
            return self.unk_id
        return self._vocab[word]

class SSANEngine:

    def __init__(self, ckpt_path, vocab_path, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1) vocab first
        self._load_vocab(vocab_path)

        # 2) opt then model
        self.opt = Opt(
            vocab_size=2500,             # 必须和训练一致
            feature_length=1024,      # 必须和训练一致
            part=6,
            caption_length_max=100    # 必须和训练一致
        )
        self.caption_length_max = self.opt.caption_length_max
        self.model = TextImgPersonReidNet(self.opt).to(self.device)
        self._load_checkpoint(ckpt_path)
        self.model.eval()

        # 3) transform（按你原项目：0.5/0.5）
        from torchvision.transforms import InterpolationMode
        self.transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

    def _load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 你的 ckpt 真实权重就在 network
        state_dict = ckpt["network"]

        # 去掉 DataParallel 前缀（如果有）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # 严格加载，确保完全一致
        self.model.load_state_dict(new_state_dict, strict=True)


    def _load_vocab(self, vocab_path):
        with open(vocab_path, "rb") as f:
            ind2word = pickle.load(f)

        self.word2ind = Word2Index(ind2word)
        self.vocab_size = len(ind2word) + 2  # 0=pad, 1..N=vocab, N+1=unk

    # --------------------------
    # 文本编码（严格对齐训练）
    # --------------------------

    def _tokenize(self, text):
        text = text.lower()

        # 保留连字符单词
        words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)

        caption = [self.word2ind(w) for w in words]
        caption_length = len(caption)

        caption = torch.tensor(caption).long()

        if caption_length < self.caption_length_max:
            padding = torch.zeros(self.caption_length_max - caption_length).long()
            caption = torch.cat([caption, padding], dim=0)
        else:
            caption = caption[:self.caption_length_max]
            caption_length = self.caption_length_max

        caption = caption.unsqueeze(0)
        caption_length = torch.tensor([caption_length]).long()
        return caption.to(self.device), caption_length.to(self.device)


    # --------------------------
    # 图像特征提取
    # --------------------------

    @torch.no_grad()
    def extract_image_features(self, img_paths, batch_size=32):

        features = []

        for i in range(0, len(img_paths), batch_size):

            batch_paths = img_paths[i:i+batch_size]
            batch_imgs = []

            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                img = self.transform(img)
                batch_imgs.append(img)

            batch_imgs = torch.stack(batch_imgs).to(self.device)

            img_global, _, _ = self.model.img_embedding(batch_imgs)

            feat = img_global.squeeze(-1).squeeze(-1)

            feat = F.normalize(feat, dim=1)

            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)


    # --------------------------
    # 文本特征提取
    # --------------------------

    @torch.no_grad()
    def extract_text_feature(self, text):

        caption_code, caption_length = self._tokenize(text)

        txt_global, _, _ = self.model.txt_embedding(
            caption_code,
            caption_length
        )

        feat = txt_global.squeeze(-1).squeeze(-1)

        feat = F.normalize(feat, dim=1)

        return feat.cpu().numpy()[0]


    # --------------------------
    # 设置图库
    # --------------------------

    def set_gallery(self, features, paths):
        self.gallery_features = features.astype(np.float32)
        self.gallery_paths = paths


    # --------------------------
    # 搜索
    # --------------------------

    def search(self, text, topk=10):

        query_feat = self.extract_text_feature(text)

        sims = self.gallery_features @ query_feat.reshape(-1, 1)
        sims = sims.reshape(-1)

        idx = np.argsort(-sims)[:topk]

        return [(self.gallery_paths[i], float(sims[i])) for i in idx]