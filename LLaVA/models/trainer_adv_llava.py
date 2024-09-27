import torch
from random import choice

def attack_pgd_untarget_outquery(image, model,  input_label, device, norm = "l_inf", epsilon= 2.0/255,
                      alpha = 0.1 , attack_iters = 3):
    ori_query = model.module.encode_images(image).data
    delta = torch.zeros_like(ori_query).to(device)

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
        loss = model.module(input_ids = input_label, images = image, labels=input_label, noise_embeds=delta).loss # adv max
        # model.module.zero_grad()
        loss.backward(retain_graph=True)
        grad = delta.grad.detach()
        d = delta[:, :, :]
        g = grad[:, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        delta.data[:, :, :] = d
        delta.grad.zero_()
    # print(attack_iters, loss)
    return model.module(input_ids = input_label, images = image, labels=input_label, noise_embeds=delta)

query_iter = [2,3,4]
query_esp = [0.8, 1.0, 1.2]

def Qattack(model, samples):
    image = samples['images']
    device = image.device
    input_label = samples['input_ids']
    attack_iters = 2 #choice(query_iter)
    epsilon = choice(query_esp)
    loss_adv = attack_pgd_untarget_outquery(image, model, input_label, device, attack_iters=attack_iters, epsilon=epsilon)
    return loss_adv

def Qattack_mix(model, samples):
    image = samples['images']
    device = image.device
    input_label = samples['input_ids']
    attack_iters = 3 #choice(query_iter)
    epsilon = choice(query_esp)
    loss_adv = attack_pgd_untarget_outquery(image, model, input_label, device, attack_iters=attack_iters, epsilon=epsilon)
    return loss_adv
