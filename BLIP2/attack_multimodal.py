import torch, json, os
from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess

def AnyNounIn(caps, nouns):
    for noun  in nouns:
        if noun in caps[0]:
            return 1
        else:
            continue
    return 0

def AllNounIn(caps, nouns):
    for noun  in nouns:
        if noun != '':
            if noun not in caps[0]:
                return 0
            else:
                continue
    return 1

def evaluate(datas):
    num_any = 0
    num_all = 0
    for ith in range(len(datas)):
        data = datas[ith]
        adv_caps = data['tgt_cap']
        tgt_word = data['out_nouns']
        num_all += AllNounIn(adv_caps, tgt_word)
        num_any += AnyNounIn(adv_caps, tgt_word)
    print("PSR_ALL:", num_all/(len(datas)))
    print("PSR_ANY:", num_all/(len(datas)))

def attack_multimodal_alignement(image, model,  text_input, device, norm = "l_inf", epsilon= 2.0/255,
                      alpha = 0.01 , attack_iters = 5,  upper_limit=1.0, lower_limit=0.0):
    delta = torch.zeros_like(model.query_tokens).to(device)     
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta.requires_grad = True
    delta.retain_grad()

    for ith in range(attack_iters):
        loss = model.forward({"image": image, "text_input": text_input, "qurey_noise": delta})
        loss['loss'].backward(retain_graph=True)
        grad = delta.grad.detach()
        d = delta[:, :, :]
        g = grad[:, :, :]

        if norm == "l_inf":
            d = torch.clamp(d - alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        delta.data[:, :, :] = d
        delta.grad.zero_()

    return delta

def test_image(image, model,tgt_text, device, attack_iters=20, epsilon=1, alpha = 0.01):
    delta =attack_multimodal_alignement(image, model, tgt_text, device=device, attack_iters=attack_iters, epsilon=epsilon, alpha = alpha)
    org_cap = model.generate({"image": image})
    tgt_cap = model.generate({"image": image,"qurey_noise": delta})
    return  org_cap, tgt_cap


if __name__ == '__main__':
    # setup device to use
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

    test_file = '/media/disk/01drive/01congzhang/git23/LAVIS/coco/annotations/coco_karpathy_test_tgt.json'
    img_dir = '/media/disk/01drive/01congzhang/dataset/COCO/coco2014/'

    with open(test_file, 'r') as fcc_file:
        data1 = json.load(fcc_file)

    output = []
    # for ith in tqdm(range(len(data1))):
    for ith in tqdm(range(10)):
        tgt_text = data1[ith]['tgt_caps']
        img_name = data1[ith]['image']
        in_nouns = data1[ith]['in_nouns']
        out_nouns = data1[ith]['out_nouns']
        
        img_path  = os.path.join(img_dir, img_name)
        raw_image = Image.open(img_path).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        org_cap, tgt_cap = test_image(image, model, [tgt_text], device, attack_iters=20, epsilon=1.0,alpha = 0.01)
        output.append({'image':img_name, "org_cap":org_cap, "tgt_cap":tgt_cap, "in_nouns":in_nouns, "out_nouns":out_nouns})
      

    evaluate(output)
  


