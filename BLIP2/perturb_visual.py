import torch
from PIL import Image
from torchvision.utils import save_image
from lavis.models import load_model_and_preprocess
from perturb_utils import clamp, denormalize, normalize


def perturb_visual_alignment(image, model,  text_input, device, norm = "l_inf", epsilon_q=1, epsilon_img=12/255,
                      alpha_q = 0.1 ,alpha_img = 1/255 , attack_iters = 5,  upper_limit=1.0, lower_limit=0.0):

    ori_query = model.extract_token({"image":image})
    delta_q = torch.zeros_like(model.query_tokens).to(device)
    delta_img = torch.zeros_like(image).to(device)
    X = denormalize(image, device)

    upper_limit = torch.tensor(upper_limit).to(device)
    lower_limit = torch.tensor(lower_limit).to(device)
    lossMSE = torch.nn.MSELoss()

    if norm == "l_inf":
        delta_img.uniform_(-epsilon_img, epsilon_img)
        delta_q.uniform_(-epsilon_q, epsilon_q)
    elif norm == "l_2":
        delta_img.normal_()
        d_flat = delta_img.view(delta_img.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta_img.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta_img *= r / n * epsilon_img

        delta_q.normal_()
        d_flat = epsilon_q.view(delta_q.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta_q.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta_q *= r / n * epsilon_q
    else:
        raise ValueError

    delta_img = clamp(delta_img, lower_limit - X, upper_limit - X)
    delta_img.requires_grad = True
    delta_img.retain_grad()
    delta_q.requires_grad = True
    delta_q.retain_grad()

    x_adv = image+delta_img
    for ith in range(attack_iters):
        loss_q = model.forward({"image": x_adv, "text_input": text_input, "qurey_noise": delta_q})
        loss_q['loss'].backward(retain_graph=True)
        grad_q = delta_q.grad.detach()
        d_q = delta_q[:, :, :]
        g_q = grad_q[:, :, :]
        if norm == "l_inf":
            d_q = torch.clamp(d_q - alpha_q * torch.sign(g_q), min=-epsilon_q, max=epsilon_q)# traget
            # d_q = torch.clamp(d_q + alpha_q * torch.sign(g_q), min=-epsilon_q, max=epsilon_q) # untarget
        delta_q.data[:, :, :] = d_q
        delta_q.grad.zero_()

        x_adv = normalize(X+ delta_img, device)
        refine_token = model.extract_token({"image":x_adv})
        loss_img = lossMSE(refine_token-ori_query, delta_q)
        loss_img.backward(retain_graph=True)
        grad_img = delta_img.grad.detach()
        d_img = delta_img[:, :, :, :]
        g_img = grad_img[:, :, :, :]
        if norm == "l_inf":
            d_img = torch.clamp(d_img + alpha_img * torch.sign(g_img), min=-epsilon_img, max=epsilon_img) # mse
        d_img = clamp(d_img, lower_limit - X.data, upper_limit - X.data)
        delta_img.data[:, :, :, :] = d_img
        delta_img.grad.zero_()

    return delta_img,delta_q


def process_image(image, model, raw_text,img_sz,attack_iters=30, epsilon_q=1.5, epsilon_img=16/255):
    delta_img, query_nosie = perturb_visual_alignment(image, model,  raw_text, device=device, attack_iters=attack_iters, epsilon_q=epsilon_q, epsilon_img=epsilon_img)
    adv_img =  denormalize(image, device) + delta_img
    return adv_img


if __name__ == '__main__':

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

    img_path = 'demo.png'
    raw_text = ['a cat is sitting in a box on the floor']
    raw_image = Image.open(img_path).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


    adv_img = process_image(image, model,raw_text, raw_image.size, attack_iters=100, epsilon_q=2.0, epsilon_img=8/255)
    save_image(adv_img.squeeze(0), "advImg.png")
