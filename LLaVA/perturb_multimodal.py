import torch, json, os
from PIL import Image
from tqdm import tqdm
from llava.conversation import conv_templates 
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def perturb_multimodal(image, model,  input_label, device, norm = "l_inf", epsilon_q=10, epsilon_img=8/255,
                      alpha_q = 1 ,alpha_img = 1/255 , attack_iters = 5,  upper_limit=1.0, lower_limit=0.0):
    ori_query = model.encode_images(image).data
    delta_q = torch.zeros_like(ori_query).to(device) 
    if norm == "l_inf":
        delta_q.uniform_(-epsilon_q, epsilon_q)
    elif norm == "l_2":
        delta_q.normal_()
        d_flat = epsilon_q.view(delta_q.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta_q.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta_q *= r / n * epsilon_q
    else:
        raise ValueError

    delta_q.requires_grad = True
    delta_q.retain_grad()
    for ith in range(attack_iters):
        loss_q = model(input_ids = input_label, images = image, labels=input_label, noise_embeds=delta_q).loss 
        loss_q.backward(retain_graph=True)
        grad_q = delta_q.grad.detach()
        d_q = delta_q[:, :, :]
        g_q = grad_q[:, :, :]
        d_q = torch.clamp(d_q - alpha_q * torch.sign(g_q), min=-epsilon_q, max=epsilon_q)# traget
        delta_q.data[:, :, :] = d_q
        delta_q.grad.zero_()

    return delta_q


def inference_caption(input_ids, image_tensor, image, delta_q=None):
    with torch.inference_mode():
        if delta_q is not None:
            output_ids = model.generate(input_ids,images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],do_sample=True,temperature=0.2, noise_embeds=delta_q,
                top_p=None, num_beams=1, max_new_tokens=1024,use_cache=True)
        else:
            output_ids = model.generate(input_ids,images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],do_sample=True,temperature=0.2,
                top_p=None, num_beams=1, max_new_tokens=1024,use_cache=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs



if __name__ == '__main__':
    # setup device to use
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, context_len = load_pretrained_model('/media/disk/01drive/01congzhang/modelslllm/llava-v1.5-7b/',None, 'llava-v1.5-7b')
    test_file = 'data/coco_karpathy_test_tgt.json'
    img_dir = '/media/disk/01drive/01congzhang/dataset/COCO/'

    with open(test_file, 'r') as fcc_file:
        data1 = json.load(fcc_file)

    output = []
    for ith in tqdm(range(len(data1))):
        tgt_text = data1[ith]['tgt_caps']
        img_name = data1[ith]['image']
        in_nouns = data1[ith]['in_nouns']
        out_nouns = data1[ith]['out_nouns']

        img_path  = os.path.join(img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

        qs = 'Describe the image concisely.'
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        conv1 = conv_templates['llava_v1'].copy()
        conv1.append_message(conv1.roles[0], qs)
        conv1.append_message(conv1.roles[1], tgt_text)
        input_text = conv1.get_prompt()
        input_ids_tgt = tokenizer_image_token(input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
       
        try:
            delat_m = perturb_multimodal(image_tensor, model, input_ids_tgt, device, epsilon_q= 2, alpha_q =0.1 , attack_iters = 20)
            org_cap = inference_caption(input_ids, image_tensor[0], image)
            tgt_cap = inference_caption(input_ids, image_tensor[0], image, delta_q=delat_m)
            output.append({'image':img_name, "org_cap":org_cap, "tgt_cap":tgt_cap, "in_nouns":in_nouns, "out_nouns":out_nouns})
        except:
            output.append({'image':img_name, "org_cap":'zc', "tgt_cap":'zc', "in_nouns":in_nouns, "out_nouns":out_nouns})


