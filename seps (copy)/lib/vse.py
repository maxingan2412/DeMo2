import torch
import torch.nn as nn
import torch.nn.init
import lib.utils as utils
import logging

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import loss_select

from lib.cross_net import CrossSparseAggrNet_v2

logger = logging.getLogger(__name__)


class VSEModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.img_enc = get_image_encoder(opt)
        self.txt_enc = get_text_encoder(opt)


        self.criterion = loss_select(opt, loss_type=opt.loss)

        # iteration
        self.Eiters = 0

        # 稀疏 + 聚合模块（Patch Slimming + 跨模态打分核心）
        self.cross_net = CrossSparseAggrNet_v2(opt)

    def freeze_backbone(self):
        self.img_enc.freeze_backbone()
        self.txt_enc.freeze_backbone()

    def unfreeze_backbone(self):
        self.img_enc.unfreeze_backbone()
        self.txt_enc.unfreeze_backbone()

    def set_max_violation(self, max_violation=True):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    # Compute the image and caption embeddings
    def forward_emb(self, images, captions, lengths, long_captions=None, long_lengths=None):
       
        images = images.cuda()
        img_emb = self.img_enc(images)

        # compute caption embs
        captions = captions.cuda()
        lengths = lengths.cuda()
        cap_emb = self.txt_enc(captions, lengths)

        long_captions = long_captions.cuda() if long_captions is not None else None
        long_lengths = long_lengths.cuda() if long_lengths is not None else None
        long_cap_emb = self.txt_enc(long_captions, long_lengths) if long_captions is not None else None
        
        if long_captions is not None:
            return img_emb, cap_emb, lengths, long_cap_emb, long_lengths
        else:
            return img_emb, cap_emb, lengths
    
    # compute the similarity on cross-attention interaction
    def forward_sim(self, img_embs, cap_embs, cap_lens, long_cap_embs=None, long_cap_lens=None):
        
        sims = self.cross_net(img_embs, cap_embs, cap_lens, long_cap_embs, long_cap_lens)

        return sims

    # One training step given images and captions
    def forward(self, images, captions, lengths, img_ids=None, warmup_alpha=1., long_captions=None, long_lengths=None):

        self.Eiters += 1
      
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        
        long_cap_emb = self.txt_enc(long_captions, long_lengths)
        
        new_img_emb = img_emb
        new_cap_emb = cap_emb
        new_long_cap_emb = long_cap_emb
        
        # get all samples for compute loss function
        if self.opt.multi_gpu:
            lengths = utils.concat_all_gather(lengths, keep_grad=False)
            img_ids = utils.concat_all_gather(img_ids, keep_grad=False)
            
            if long_lengths is not None:
                long_lengths = utils.concat_all_gather(long_lengths, keep_grad=False)
            
            max_len = int(lengths.max())
            if max_len > new_cap_emb.shape[1]:
                pad_emb = torch.zeros(new_cap_emb.shape[0], max_len - new_cap_emb.shape[1], new_cap_emb.shape[2], ).to(new_cap_emb.device)
                new_cap_emb = torch.cat([new_cap_emb, pad_emb], dim=1)
            
            if new_long_cap_emb is not None:
                max_long_len = int(long_lengths.max())
                if max_long_len > new_long_cap_emb.shape[1]:
                    pad_long_emb = torch.zeros(new_long_cap_emb.shape[0], max_long_len - new_long_cap_emb.shape[1], new_long_cap_emb.shape[2], ).to(new_long_cap_emb.device)
                    new_long_cap_emb = torch.cat([new_long_cap_emb, pad_long_emb], dim=1)
                
                new_long_cap_emb = utils.all_gather_with_grad(new_long_cap_emb)
            
            new_img_emb = utils.all_gather_with_grad(new_img_emb)
            new_cap_emb = utils.all_gather_with_grad(new_cap_emb)

        # compute similarity matrix
        improved_sims, score_mask_all = self.forward_sim(new_img_emb, new_cap_emb, lengths, new_long_cap_emb, long_lengths)
        # basic alignment loss
        align_loss = self.criterion(new_img_emb, new_cap_emb, img_ids, improved_sims) * warmup_alpha
        # ratio_loss 约束保留 patch 的比例接近 sparse_ratio 超参
        ratio_loss = (score_mask_all.mean() - self.opt.sparse_ratio) ** 2
        loss = align_loss + self.opt.ratio_weight * ratio_loss
        return loss, None


# optimizer init
def create_optimizer(opt, model):

    # Set up the lr for different parts of the VSE model
    decay_factor = 1e-4  
    cross_lr_rate = 1.0
        
    # bert params
    all_text_params = list(model.txt_enc.parameters())
    bert_params = list(model.txt_enc.bert.parameters())
    bert_params_ptr = [p.data_ptr() for p in bert_params]
    text_params_no_bert = list()

    for p in all_text_params:
        if p.data_ptr() not in bert_params_ptr:
            text_params_no_bert.append(p)

    # bert   
    params_list = [
        {'params': text_params_no_bert, 'lr': opt.learning_rate},
        {'params': bert_params, 'lr': opt.learning_rate * 0.1},
    ]

    # vit
    params_list += [
        {'params': model.img_enc.visual_encoder.parameters(), 'lr': opt.learning_rate * 0.1},
        {'params': model.img_enc.vision_proj.parameters(), 'lr': opt.learning_rate},
    ]

    # cross-moadl alignment 
    params_list += [
        {'params': model.cross_net.parameters(), 'lr': opt.learning_rate * cross_lr_rate},
        {'params': model.criterion.parameters(), 'lr': opt.learning_rate},
    ]   
  
    optimizer = torch.optim.AdamW(params_list, lr=opt.learning_rate, weight_decay=decay_factor)
    
    return optimizer


if __name__ == '__main__':

    pass
