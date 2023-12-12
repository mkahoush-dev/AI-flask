from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
def run(title, length):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")
    blogpost_title = title
    prompt = f'This is a blog post on {blogpost_title} \n {blogpost_title} is very important '
    input_ids = tokenizer(prompt, return_tensors="pt")
    sample = model.generate(**input_ids, max_length=length, num_beams = 2, num_beam_groups = 2, top_k=1, temperature=0.9, repetition_penalty = 2.0, diversity_penalty = 0.9)
    return tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
# run("The Essentials Of Leadership", 100)