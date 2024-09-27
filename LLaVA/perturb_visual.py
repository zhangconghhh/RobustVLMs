import argparse, torch, os
from PIL import Image
from torchvision.utils import save_image
from attack_utils import clamp, denormalize, normalize
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def perturb_visual_alignment(image, model,  input_label, device, norm = "l_inf", epsilon_q=10, epsilon_img=8/255,
                      alpha_q = 1 ,alpha_img = 1/255 , attack_iters = 5, attack_iters_img=100):
    ori_query = model.encode_images(image).data
    delta_q = torch.zeros_like(ori_query).to(device) 
    delta_img = torch.zeros_like(image).to(device) 
    lossMSE = torch.nn.MSELoss()
    delta_img.uniform_(-epsilon_img, epsilon_img)
    delta_q.uniform_(-epsilon_q, epsilon_q)
    img_lower_limit = normalize(torch.zeros_like(image), device) - image
    img_upper_limit = normalize(torch.ones_like(image), device) - image
    delta_img.requires_grad = True
    delta_img.retain_grad()
    delta_q.requires_grad = True
    delta_q.retain_grad()
    x_adv = image + delta_img

    for ith in range(20):
        loss_q = model(input_ids = input_label, images = image, labels=input_label, noise_embeds=delta_q).loss # adv max
        loss_q.backward(retain_graph=True)
        grad_q = delta_q.grad.detach()
        d_q = delta_q[:, :, :]
        g_q = grad_q[:, :, :]
        d_q = torch.clamp(d_q + alpha_q * torch.sign(g_q), min=-epsilon_q, max=epsilon_q)
        delta_q.data[:, :, :] = d_q
        delta_q.grad.zero_()
    for ith in range(attack_iters):
        x_adv = image + delta_img
        refine_token = model.model.mm_projector(model.model.vision_tower.vision_tower(x_adv, output_hidden_states=True).hidden_states[-2][:, 1:].half())
        loss_img = lossMSE(refine_token-ori_query, delta_q) # update use -
        loss_img.backward(retain_graph=True)
        grad_img = delta_img.grad.detach()
        d_img = delta_img[:, :, :, :]
        g_img = grad_img[:, :, :, :]
        if norm == "l_inf":
            d_img = torch.clamp(d_img + alpha_img * torch.sign(g_img), min=-epsilon_img, max=epsilon_img)
        d_img = clamp(d_img, img_lower_limit, img_upper_limit)
        delta_img.data[:, :, :, :] = d_img
        delta_img.grad.zero_()
    adv_img =  image + delta_img

    return adv_img, delta_q


def eval_model(args):
    disable_torch_init()
    device = torch.device("cuda:1")
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=device)
    qs = 'Describe the image concisely.'+"\nAnswer the question using a single sentence."
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], args.img_caption)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
 
    image = Image.open(args.img_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    adv_image_tensor, delta_q = perturb_visual_alignment(image_tensor, model,  input_ids, device, epsilon_q=1, epsilon_img=32/255, alpha_q=0.1, alpha_img=1/255, attack_iters=100, attack_iters_img=100)
    save_image(denormalize(adv_image_tensor, device).squeeze(0), 'advImg.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/media/disk/01drive/models/llava-v1.5-7b/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--img-path", type=str, default='./demo.png')
    parser.add_argument("--img-caption", type=str, default='a cat is sitting in a box on the floor')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
