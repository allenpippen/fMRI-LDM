from transformers import CLIPTokenizer
import torch
def get_tokenizer():
    tokenizer = CLIPTokenizer.from_pretrained(
        './weights/diffsion_from_scratch.params', subfolder='tokenizer')

    return tokenizer

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = get_tokenizer()
    print(tokenizer)

    pos = tokenizer("Group is Autism Spectrum Disorder. Age is 26 years old. Gender is Female.",
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(device)
    neg = tokenizer('Group is Healthy Control. Age is 14 years old. Gender is Male.',
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(device)
    tmp = tokenizer('Group is Autism Spectrum Disorder.',
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(device)
    print(pos)
    print(neg)
    print(tmp)