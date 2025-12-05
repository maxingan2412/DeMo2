import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, SwinModel, ViTModel
import logging


logger = logging.getLogger(__name__)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def get_text_encoder(opt):
    txt_enc = EncoderText_BERT(opt)   
    return txt_enc


def get_image_encoder(opt):
    img_enc = VisionTransEncoder(opt)
    return img_enc


# ViT encoder
class VisionTransEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        if 'swin' in opt.vit_type:                           
            if 'swin_384' in opt.vit_type:
                # img_res 384 * 384, 12*12 patch (window_size=12, patch_size=4)
                try:
                    self.visual_encoder = SwinModel.from_pretrained("microsoft/swin-base-patch4-window12-384")
                    opt.num_patches = 144  # (384/4/12)^2 * (12*12) = 12*12 = 144
                    print('swin-384 model with 12x12 patches')
                except:
                    print("Warning: swin-384 model not found, using swin-224 with 384 input")
            elif 'swin_224' in opt.vit_type or opt.vit_type == 'swin':
                # img_res 224 * 224, 7*7 patch
                self.visual_encoder = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
                opt.num_patches = 49
                print('swin-224 model with 7x7 patches')
        #  ViT model
        else:              
            if 'vit_384' in opt.vit_type:
                # img_res 384 * 384, 24*24 patch (patch_size=16)
                self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-384")
                opt.num_patches = 576  # (384/16)^2 = 24*24 = 576
                print('vit-384 model with 24x24 patches')
            elif 'vit_224' in opt.vit_type or opt.vit_type == 'vit':
                # img_res 224 * 224, 14*14 patch (patch_size=16)
                self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
                opt.num_patches = 196  # (224/16)^2 = 14*14 = 196
                print('vit-224 model with 14x14 patches')   

        # dimension transform
        if opt.embed_size == self.visual_encoder.config.hidden_size:
            self.vision_proj = nn.Identity()
        else:
            self.vision_proj = nn.Linear(self.visual_encoder.config.hidden_size, opt.embed_size)            

    def forward(self, images):
    
        # (B, L_v, C_hidden) ViT/Swin 输出的 patch token（若存在则包含 CLS）
        img_feats = self.visual_encoder(images).last_hidden_state 

        # the dimension transform
        # (B, L_v, C)
        img_feats = self.vision_proj(img_feats)

        return img_feats  
        
    def freeze_backbone(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.visual_encoder.parameters():  
            param.requires_grad = True     


# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    def __init__(self, opt):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt
        self.embed_size = opt.embed_size
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # self.tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
        # self.bert = BertModel.from_pretrained(opt.bert_path)
        
        if opt.embed_size == self.bert.config.hidden_size:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.bert.config.hidden_size, opt.embed_size)

    def forward(self, x, lengths):

        # Embed word ids to vectors
        # pad 0 for redundant tokens in previous process
        bert_attention_mask = (x != 0).float()


        # N = max_cap_lengths, D = 768；输出逐 token 嵌入（不做池化）
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]  # B x N x D

        # B x N x embed_size
        cap_emb = self.fc(bert_emb)
        
        return cap_emb        

    def freeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.bert.parameters():  
            param.requires_grad = True  


if __name__ == '__main__':

    pass
