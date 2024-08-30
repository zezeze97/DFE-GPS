import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import re
from transformers import TextStreamer

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
    # 使用正则表达式查找各个部分
    patterns = {
        'construction_cdl': r'(?:The )?(?:calibrate )?construction_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
        'image_cdl': r'(?:The )?(?:calibrate )?image_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
        'text_cdl': r'(?:The )?text_cdl(?: is)?:\n(.*?)(?=\n(?:The )?\w+_cdl is:|\n(?:The )?\w+_cdl:|\nSolution is:|\Z)',
        'goal_cdl': r'(?:The )?goal_cdl(?: is)?:\n(.*?)(?=\n(?:The )?\w+_cdl is:|\n(?:The )?\w+_cdl:|\nSolution is:|\Z)'
    }
    
    results = {}
    
    # 优先匹配包含"calibrate"的版本
    for key, pattern in patterns.items():
        pattern = pattern.replace("(?:calibrate )?", "(?:calibrate )")
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            results[key] = match.group(1).strip()
        else:
            # 如果未找到包含"calibrate"的版本，尝试匹配不含"calibrate"的版本
            pattern = pattern.replace("(?:calibrate )", "(?:calibrate )?")
            match = re.search(pattern, input_string, re.DOTALL)
            if match:
                results[key] = match.group(1).strip()
    
    return results





def cascade_prediction(formalization_tokenizer, formalization_model,
                       reasonging_tokenizer, reasoning_model, img_path, qs, qa_mode, consCDL=None, imgCDL=None, temperature=0, num_beams=1, top_p=None, device='cuda'):
    '''
    qs是原始问题, 不包含任何格式
    qa_mode: q2ans, q2cdl_ans, q_cdl2ans, q_predcdl2ans, q_predcdl2cdl_ans
    
    '''
    pred_consCDL = ''
    pred_imgCDL = ''
    image = Image.open(img_path).convert('RGB')
    assert formalization_model.config.image_aspect_ratio == reasoning_model.config.image_aspect_ratio
    
    # 需要模型预测CDL
    if qa_mode in ['q_predcdl2ans', 'q_predcdl2cdl_ans']:
        # 识别的时候使用先识别，再矫正指令
        fomalization_prompt = 'Based on the image, first describe what you see in the figure, then predict the construction_cdl and image_cdl and calibrate it.'
        text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{fomalization_prompt}<|im_end|>\n<|im_start|>assistant\n"
        formalization_input_ids = tokenizer_image_token(text, formalization_tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
        formalization_image_tensor = formalization_model.process_images([image], formalization_model.config).to(dtype=formalization_model.dtype, device=device)
        formalization_streaner = TextStreamer(formalization_tokenizer, skip_prompt=True, skip_special_tokens=True)
        # 进行结构识别
        print('Start Diagram Formalization....')
        with torch.inference_mode():
            formalization_output_ids = formalization_model.generate(
                formalization_input_ids,
                images=formalization_image_tensor.unsqueeze(0).to(dtype=formalization_model.dtype, device='cuda', non_blocking=True),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=3500, # 2048
                eos_token_id=formalization_tokenizer.eos_token_id,
                repetition_penalty=None,
                use_cache=True,
                streamer=formalization_streaner
                )
        formalization_input_token_len = formalization_input_ids.shape[1]
        n_diff_input_output = (formalization_input_ids != formalization_output_ids[:, :formalization_input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        formalization_outputs = formalization_tokenizer.batch_decode(formalization_output_ids[:, formalization_input_token_len:], skip_special_tokens=True)[0]
        formalization_outputs = formalization_outputs.strip()
        cdl_dict = parse_cdl(formalization_outputs)
        if 'construction_cdl' in cdl_dict.keys():
            pred_consCDL = cdl_dict['construction_cdl']
        if 'image_cdl' in cdl_dict.keys():
            pred_imgCDL = cdl_dict['image_cdl']
    
    # 开始模型推理
    # 构建问题
    qs_map = {'q2ans': f'Using the provided geometric image and question, give a detailed step-by-step solution.\nThe question is:\n{qs}',
     'q2cdl_ans': f'Using the provided geometric image and question, first predict the construction_cdl and image_cdl. Then, give a detailed step-by-step solution.\nThe question is:\n{qs}',
     'q_cdl2ans': f'Using the provided geometric image, construction_cdl, image_cdl, and question, give a detailed step-by-step solution.\nThe construction_cdl is:\n{consCDL}\nThe image_cdl is:\n{imgCDL}\nThe question is:\n{qs}',
     'q_predcdl2ans': f'Using the provided geometric image, construction_cdl, image_cdl, and question, give a detailed step-by-step solution. Note that there may be minor errors in the construction_cdl and image_cdl.\nThe construction_cdl is:\n{pred_consCDL}\nThe image_cdl is:\n{pred_imgCDL}\nThe question is:\n{qs}',
     'q_predcdl2cdl_ans': f'Using the provided geometric image and the possibly erroneous construction_cdl and image_cdl, first calibrate the construction_cdl and image_cdl, then give a detailed step-by-step solution to the question.\nThe initial construction_cdl is:\n{pred_consCDL}\nThe initial image_cdl is:\n{pred_imgCDL}\nThe question is:\n{qs}'}
    
    prompt = qs_map[qa_mode]
    text = f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer_image_token(text, reasonging_tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
    image_tensor = reasoning_model.process_images([image], reasoning_model.config).to(dtype=reasoning_model.dtype, device=device)
    streamer = TextStreamer(reasonging_tokenizer, skip_prompt=True, skip_special_tokens=True)
    print('Start Reasoning...')
    with torch.inference_mode():
        output_ids = reasoning_model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=reasoning_model.dtype, device='cuda', non_blocking=True),
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            top_k=None,
            num_beams=num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=3500, # 2048
            eos_token_id=reasonging_tokenizer.eos_token_id,
            repetition_penalty=None,
            use_cache=True,
            streamer=streamer
            )
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = reasonging_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    
    # print(f"CotAns is:\n{outputs}")
    
            
            
def main(args):
    # img_path = args.img_path
    # qs = 'As shown in the diagram, AE/AB=1/4, M is the midpoint of segment AC, BE is parallel to CP, EA is parallel to CP. Find the ratio of the length of line BC to the length of line CD.'
    qa_mode = args.qa_mode
    # 加载识别模型和推理模型
    
    # disable some warnings
    # transformers.logging.set_verbosity_error()
    # transformers.logging.disable_progress_bar()
    # warnings.filterwarnings('ignore')
    # set device
    torch.set_default_device('cuda')  # or 'cuda'

    formalization_model = AutoModelForCausalLM.from_pretrained('NaughtyDog97/DiagramFormalizer',
                                                                torch_dtype=torch.float16, 
                                                                device_map='auto',
                                                                trust_remote_code=True)
    formalization_tokenizer = AutoTokenizer.from_pretrained('NaughtyDog97/DiagramFormalizer',
                                                            use_fast=True,
                                                            padding_side="right",
                                                            trust_remote_code=True)
    


    reasoning_model = AutoModelForCausalLM.from_pretrained('NaughtyDog97/DFE-GPS-34B', # or 9B
                                                            torch_dtype=torch.float16, 
                                                            device_map='auto',
                                                            trust_remote_code=True)
    reasoning_tokenizer = AutoTokenizer.from_pretrained('NaughtyDog97/DFE-GPS-34B', # or 9B
                                                        use_fast=False,
                                                        trust_remote_code=True)
    
    while True:
        try:
            print('Please enter the image path.')
            img_path = input()
            print('Please enter the question.')
            qs = input()
        except EOFError:
            img_path = ""
            qs = ""
        if not qs or not img_path:
            print("exit...")
            break
        cascade_prediction(formalization_tokenizer, formalization_model,
                        reasoning_tokenizer, reasoning_model, img_path, qs, qa_mode, consCDL=args.consCDL, imgCDL=args.imgCDL, temperature=0, num_beams=1, top_p=None, device='cuda')
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument('--img-path', type=str, default='data_backup/formalgeo7k/formalgeo7k_v2/diagrams/4927.png')
    parser.add_argument('--qa-mode', type=str, default='q_predcdl2cdl_ans') # q2ans, q2cdl_ans, q_cdl2ans, q_predcdl2ans, q_predcdl2cdl_ans
    parser.add_argument('--consCDL', type=str, default=None)
    parser.add_argument('--imgCDL', type=str, default=None)
    args = parser.parse_args()
    main(args)