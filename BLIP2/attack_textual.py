import torch, pdb, json
import numpy as np
from tqdm import tqdm
from PIL import Image
import requests
import torch.nn.functional as F
import torch.nn as nn
from lavis.models import load_model_and_preprocess
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from torchvision.utils import save_image
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions



# setup device to use
device = torch.device("cuda:3") if torch.cuda.is_available() else "cpu"
model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
# Jfile = '/media/disk/01drive/01congzhang/git23/LAVIS/coco/annotations/coco_karpathy_test.json'
# # Jfile = '/media/disk/01drive/01congzhang/git23/LAVIS/coco/annotations/coco_karpathy_val.json'
# with open(Jfile, 'r') as fcc_file:
#     data = json.load(fcc_file)

# for ith in tqdm(range(len(data))):
# # 500 -> 66h, 100 ->10
# for ith in tqdm(range(1300, 1400)):
# # if True:
#     img_name = data[ith]['image']
#     if True:
#         raw_text = [data[ith]['caption'][0]]
#     # for cth in range(len(data[ith]['caption'])):
    #     raw_text = [data[ith]['caption'][cth]]
        # raw_text = ['A man with a red helmet on a small moped on a dirt road. ']
        # img_name = 'val2014/COCO_val2014_000000391895.jpg'
        # img_path = '/media/disk/01drive/01congzhang/dataset/COCO/coco2014/' + img_name

img_path = 'demo.png'
raw_text = 'a cat is sitting in a box on the floor'
raw_image = Image.open(img_path).convert('RGB')
img_sz = raw_image.size
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# pdb.set_trace()
caption = text_processors["eval"](raw_text)  # itc=0.46 itm=0.99

with model.maybe_autocast():
    image_embeds = model.ln_vision(model.visual_encoder(image)).float()
image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)

text = model.tokenizer(caption, truncation=True, max_length=32, return_tensors="pt").to(image.device)
embedding_output = model.Qformer.bert.embeddings.word_embeddings(text.input_ids).detach() # [1,17.768] no position embed
embed_weights = model.Qformer.bert.get_input_embeddings().weight.data

sim_min =1.0
adv_id_len = 2
attack_iter = 100
adv_token_embed = torch.tensor(embed_weights.mean(axis=0).repeat(adv_id_len, 1), requires_grad=True)
optim = torch.optim.Adam([adv_token_embed], lr=0.01) # 1000: 0.01-0.1536 0.001-0.1872; 100: 0.01-0.2321 0.001-0.2707


for jth in range(attack_iter):
    with torch.no_grad():
        diff = torch.sum((adv_token_embed.data.unsqueeze(1) - embed_weights.unsqueeze(0)) ** 2, dim=-1) # [4, 1, 768] - [1, 32855, 768] = [4, 32855, 768] -> [4, 32855]
        token_idx= diff.argmin(dim=1)
        q_adv_token_embed = embed_weights[token_idx]#[-1-adv_id_len:-1]]

    q_adv_token_embed = q_adv_token_embed.data - adv_token_embed.data + adv_token_embed
    adv_ids_new = torch.cat([text.input_ids[:,:-1], token_idx.unsqueeze(0), text.input_ids[:,-1].unsqueeze(0)], dim=1)
    full_embed = torch.cat([embedding_output[:,:-1,:], q_adv_token_embed.unsqueeze(0), embedding_output[:,-1,:].unsqueeze(1)], dim=1) # 把中间那几个替换了


    embedding_output_adv = model.Qformer.bert.embeddings.forward_adv(position_ids=None,query_embeds=full_embed,past_key_values_length=0,) # [1,17.768]
    extended_attention_mask = model.Qformer.bert.get_extended_attention_mask(torch.ones(adv_ids_new.shape), full_embed.size()[:-1], device, False).to(embedding_output_adv.device)
    head_mask = model.Qformer.bert.get_head_mask(None, 12)
    encoder_outputs = model.Qformer.bert.encoder(
        embedding_output_adv,attention_mask=extended_attention_mask,head_mask=head_mask,encoder_hidden_states=None,encoder_attention_mask=None,
        past_key_values=None,use_cache=None,output_attentions=False,output_hidden_states=False,return_dict=True,query_length=0,)
    sequence_output = encoder_outputs[0]
    text_output = BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output,pooler_output=None,past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,attentions=encoder_outputs.attentions,cross_attentions=encoder_outputs.cross_attentions,)

    text_feat = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1) #[1,768] -> [1,256]
    query_output = model.Qformer.bert(query_embeds=query_tokens, encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True)
    image_feats = F.normalize(model.vision_proj(query_output.last_hidden_state), dim=-1) # [1,32,768] -> [1,32,256]
    sim_adv =torch.max(torch.bmm(image_feats, text_feat.unsqueeze(-1)), dim=1)[0]

    if sim_adv < sim_min:
        sim_min = sim_adv
        adv_ids = adv_ids_new

    optim.zero_grad()
    sim_adv.backward(retain_graph=True)
    optim.step()
adv_cap = model.tokenizer.decode(adv_ids[0], skip_special_tokens=True)
# data[ith]['caption'] = [adv_cap]
    # data[ith]['caption'][cth] = adv_cap

# print(sim_min)

# calculate scores
itc_score = model({"image": image, "text_input": caption}, match_head='itc')
itm_output = model({"image": image, "text_input": caption}, match_head="itm")
itm_scores = torch.nn.functional.softmax(itm_output, dim=1)[:,1]


adv = text_processors["eval"](adv_cap)
itc_score_adv = model({"image": image, "text_input": adv}, match_head='itc')
itm_output_adv = model({"image": image, "text_input": [adv_cap]}, match_head="itm")
itm_scores_adv = torch.nn.functional.softmax(itm_output_adv, dim=1)[:,1]
print("After teatual aligenmet perturbation the ITC score decrease from ", itc_score , " to ", itc_score_adv)
print("After teatual aligenmet perturbation the ITM score decrease from ", itm_scores , " to ", itm_scores_adv)
# print(itc_score,itm_scores,itc_score_adv,itm_scores_adv)
# pdb.set_trace()

# with open("text_attack_itc_100_1_2token.json","w") as f:
#     json.dump(data, f)
