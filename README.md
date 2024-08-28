# Diagram Formalization Enhanced Multi-Modal Geometry Problem Solver
**Dataset**: 🤗 [SynthGeo-228k](https://huggingface.co/datasets/JO-KU/SynthGeo228K) | **Diagram Formalizer**: 🤗 [Diagram Formalizer](https://huggingface.co/NaughtyDog97/DiagramFormalizer) | **Reasoning Model**: 🤗 [DFE-GPS-9B](https://huggingface.co/NaughtyDog97/DFE-GPS-9B) | 🤗 [DFE-GPS-34B](https://huggingface.co/NaughtyDog97/DFE-GPS-34B)


<p align="center">
  <img src="images/pipeline.png" alt="Alt text" width="100%" height="auto">
</p>

In this study, we introduce the Diagram Formalization Enhanced Geometry Problem Solver (DFE-GPS), a multi-modal architecture with three core components: a Diagram Formalizer, a Projection module, and a LLM. The LLM processes three types of inputs: diagram features $\mathcal{F}_{D}$ from the Diagram Encoder, formal diagram language representations (ConsCDL and ImgCDL) from the Diagram Formalizer, and natural language inputs containing problem statements and instructions. The Projection module aligns this information in the LLM's semantic space, enabling effective fusion. The LLM then refines the formal representations and generates reasoning steps for problem-solving. We used pre-trained [SigLIP-0.4B]() as the Vision Encoder, [Qwen2-0.5B-Instruct]() as the Lightweight LLM, and [Yi-1.5-Chat]() (9B or 34B) as the primary LLM. The training process is divided into three stages, all of which focus on auto-regressive generation tasks.
- **Stage 1**: The first stage focuses on training the Diagram Formalizer module, with training objective of generating formalized language descriptions corresponding to geometric diagrams. During this stage, all parameters of the Vision Encoder and part of the parameters of the Lightweight LLM (via the LoRA) are trainable to enhance the ability to extract visual features.
- **Stage 2**: The second stage emphasizes training the Projection modules, aligning vision feature $\mathcal{F}_{D}$ with the semantic space of the LLM by generating natural language descriptions and formalized language expressions for the geometric diagrams. During training, the parameters of the Diagram Encoder and LLM are frozen, with only the MLP parameters connecting the visual features and the language model being trainable.
- **Stage 3**: In the third stage, instruction fine-tuning enables the model to calibrate formalized diagram representations and solve problems. The input consists of geometric diagrams, formalized descriptions with random perturbations simulating Diagram Formalizer errors, and problem text accompanied by calibration and reasoning directives. The model learns to refine ConsCDL and ImgCDL, generating coherent natural language reasoning. During this phase, the parameters of the Diagram Encoder remain fixed, while the MLP and LLM parameters are trainable. Full parameter tuning is applied to the 9B model, whereas LoRA tuning is employed for the 34B model.

## Peformance
<p align="center">
  <img src="images/main_expr.png" alt="Alt text" width="50%" height="auto">
</p>


## Quick Start
Before running the script, install the following necessary dependencies.

```shell
pip install torch transformers==4.40.0 accelerate pillow sentencepiece
```

You can solve geometric problems using the following script. First, formalize the geometric images with the [Diagram Formalizer](https://huggingface.co/NaughtyDog97/DiagramFormalizer), and then use the multi-modal reasing model for problem-solving:

```python
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np
import re

def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def parse_cdl(input_string):
    patterns = {
        'construction_cdl': r'(?:The )?(?:calibrate )?construction_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
        'image_cdl': r'(?:The )?(?:calibrate )?image_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
        'text_cdl': r'(?:The )?text_cdl(?: is)?:\n(.*?)(?=\n(?:The )?\w+_cdl is:|\n(?:The )?\w+_cdl:|\nSolution is:|\Z)',
        'goal_cdl': r'(?:The )?goal_cdl(?: is)?:\n(.*?)(?=\n(?:The )?\w+_cdl is:|\n(?:The )?\w+_cdl:|\nSolution is:|\Z)'
    }
    
    results = {}
    for key, pattern in patterns.items():
        pattern = pattern.replace("(?:calibrate )?", "(?:calibrate )")
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            results[key] = match.group(1).strip()
        else:
            pattern = pattern.replace("(?:calibrate )", "(?:calibrate )?")
            match = re.search(pattern, input_string, re.DOTALL)
            if match:
                results[key] = match.group(1).strip()
    
    return results


# set device
device = 'cuda'  # or cpu
torch.set_default_device(device)

# create model
formalization_model = AutoModelForCausalLM.from_pretrained(
    'NaughtyDog97/DiagramFormalizer',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

formalization_tokenizer = AutoTokenizer.from_pretrained(
    'NaughtyDog97/DiagramFormalizer',
    use_fast=True,
    padding_side="right",
    trust_remote_code=True)


reason_model = AutoModelForCausalLM.from_pretrained(
    'NaughtyDog97/DFE-GPS-34B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
reason_tokenizer = AutoTokenizer.from_pretrained(
    'NaughtyDog97/DFE-GPS-34B',
    use_fase=False,
    trust_remote_code=True)



img_path = 'sample/4927.png'
image = Image.open(img_path).convert('RGB')


# formalization
prompt = 'Based on the image, first describe what you see in the figure, then predict the construction_cdl and image_cdl and calibrate it.'
text = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
input_ids = tokenizer_image_token(text, formalization_tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()

# generate
image_tensor = formalization_model.process_images([image], formalization_model.config).to(dtype=formalization_model.dtype, device=device)
with torch.inference_mode():
    output_ids = formalization_model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=1,
        max_new_tokens=3500,
        eos_token_id=formalization_tokenizer.eos_token_id,
        repetition_penalty=None,
        use_cache=True
    )[0]


respones = formalization_tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
print(f'Formalization result is\n{respones}')
cdl_info = parse_cdl(respones)
predict_consCDL = cdl_info['construction_cdl']
predict_imgCDL = cdl_info['image_cdl']



# reasoning

qs = 'As shown in the diagram, AE/AB=1/4, M is the midpoint of segment AC, BE is parallel to CP, EA is parallel to CP. Find the ratio of the length of line BC to the length of line CD.'
prompt = f'Using the provided geometric image and the possibly erroneous construction_cdl and image_cdl, first calibrate the construction_cdl and image_cdl, then give a detailed step-by-step solution to the question.\nThe initial construction_cdl is:\n{predict_consCDL}\nThe initial image_cdl is:\n{predict_imgCDL}\nThe question is:\n{qs}'
text = f'<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
input_ids = tokenizer_image_token(text, reason_tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()



# generate
image_tensor = reason_model.process_images([image], reason_model.config).to(dtype=reason_model.dtype, device=device)
with torch.inference_mode():
    output_ids = reason_model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=1,
        max_new_tokens=3500,
        eos_token_id=reason_tokenizer.eos_token_id,
        repetition_penalty=None,
        use_cache=True
    )[0]

respones = reason_tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
print(f'Reasoning steps is\n{respones}')

```