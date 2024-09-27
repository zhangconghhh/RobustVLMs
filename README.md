# Enhancing the Robustness of Vision-Language Foundation Models by Alignment Perturbation

This is the official implementation of Enhancing the Robustness of Vision-Language Foundation Models by Alignment Perturbation


## Alignment Perturbation on BLIP-2 

1. Installation the environemnt following the [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) repository.

2. Replace the LAVIS/lavis/models/blip2_models/blip2_opt.py with RobustVLMs/BLIP2/models/blip2_opt.py

3. Run the perturbation codes.
 + Multimodal Alignment Perturbation： python BLIP2/perturb_multimodal.py
 + Visual Alignment Perturbation： python BLIP2/perturb_visual.py
 + Textual Alignment Perturbation： python BLIP2/perturb_textual.py


## Alignment Perturbation on LLaVA

1. Installation the environemnt following the [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main) repository.

2. Run the perturbation codes.
 + Multimodal Alignment Perturbation： python LLaVA/perturb_multimodal.py
 + Visual Alignment Perturbation： python LLaVA/perturb_visual.py
 + Textual Alignment Perturbation： python LLaVA/perturb_textual.py


 ### Alignment Robust Training on LLaVA

 1. Complete the following code changes in the conda environment:

+  Replace anaconda3/envs/llava/lib/python3.10/site-packages/transformers/trainer.py with RobustVLMs/LLaVA/models/trainer.py

+  Add RobustVLMs/LLaVA/models/trainer_adv_llava.py to anaconda3/envs/llava/lib/python3.10/site-packages/transformers/trainer_adv_llava.py

2. Run code the finetuning code
```
sh scripts/v1_5/finetune_lora.sh
```

<!-- ## Cite the paper
If this work is helpful to you, please cite it as:</p>
``` -->

## Acknowledgements
We appreciate the wonderful base implementation of [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip), and [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main). 
