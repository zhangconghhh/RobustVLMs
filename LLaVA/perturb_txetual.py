import argparse, torch, os
from PIL import Image
from torch.nn import CrossEntropyLoss
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from transformers.modeling_outputs import CausalLMOutputWithPast
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def eval_model(args):
  
    disable_torch_init()
    device = torch.device("cuda:0")
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=device)
    image = Image.open(args.img_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

    qs =  args.img_question + "\nAnswer the question using a single word or phrase."
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    loss_fct = CrossEntropyLoss()
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    prompt_1 = conv.get_prompt()
    conv.append_message(conv.roles[1], args.img_answer)
    prompt = conv.get_prompt()
    input_ids_1 = tokenizer_image_token(prompt[:len(prompt_1)-52], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids_2 = tokenizer_image_token(prompt[len(prompt_1)-52:], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids_raw = torch.cat((input_ids_1[:,:-1],input_ids_2[:, 1:]),1)
    input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = model.prepare_inputs_labels_for_multimodal(
            input_ids_raw,  None, None, None, input_ids_raw, image_tensor.half(), None,  None )
    input_ids_ans =  tokenizer_image_token(args.img_answer, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    inputs_embeds_ans = model.model.embed_tokens(input_ids_ans)
    loss_cache =0.0
    adv_id_len = 4
    attack_iter = 100
    embed_weights = model.get_input_embeddings().weight.data.detach() 
    diff_raw = torch.sum((inputs_embeds_ans[:,0,:].unsqueeze(1).data - embed_weights.unsqueeze(0)) ** 2, dim=-1)
    _, diff_idx = torch.sort(diff_raw[0], descending=True)
    adv_token_embed = torch.tensor(embed_weights[diff_idx[:adv_id_len]].cuda().to(device), requires_grad=True)
   

    for jth in range(attack_iter):
        with torch.no_grad():
            diff = torch.sum((adv_token_embed.data.unsqueeze(1) - embed_weights.unsqueeze(0)) ** 2, dim=-1) 
            token_idx= diff.argmin(dim=1)
            q_adv_token_embed = embed_weights[token_idx]
        q_adv_token_embed = q_adv_token_embed.data - adv_token_embed.data + adv_token_embed
        inputs_embeds_adv = torch.cat((inputs_embeds[:,:input_ids_1.shape[1]+576-52], q_adv_token_embed.unsqueeze(0), inputs_embeds[:,input_ids_1.shape[1]+576-52:]), 1)
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds_adv, use_cache=None, output_attentions=False, output_hidden_states=False,  return_dict=True,)
        logits = model.lm_head(outputs[0]).float() 
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous() 
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, len(tokenizer.get_vocab())) 
        shift_labels = shift_labels.view(-1) #[638]
        shift_labels = torch.cat((shift_labels[:input_ids_1.shape[1]+576], torch.ones(adv_id_len, dtype=torch.long).fill_(-100).cuda(), shift_labels[input_ids_1.shape[1]+576:]))
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        loss_z = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,).loss
        if loss_z > loss_cache:
            loss_cache = loss_z
            adv_ids = token_idx

        # optim.zero_grad()
        loss.backward(retain_graph=True)
        grad = adv_token_embed.grad.detach()
        adv_token_embed.data =  adv_token_embed.data + 0.01 * torch.sign(grad)

    adv_que = tokenizer.decode(adv_ids)
    return  qs[:-51] + adv_que
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/media/disk/01drive/01congzhang/modelslllm/llava-v1.5-7b/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--img-path", type=str, default='./demo.png')
    parser.add_argument("--img-question", type=str, default='where is the cat?')
    parser.add_argument("--img-answer", type=str, default='in the box')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    adv_question = eval_model(args)
 