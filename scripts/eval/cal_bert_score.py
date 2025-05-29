#计算answer和标准output之间的bertscore
import bert_score
import numpy as np
import torch
from tqdm import tqdm
# def calculate_bert_score(candidate, reference):
#     """
#     计算答案和标准答案之间的 BERTScore
#     :param candidate: 候选答案
#     :param reference: 标准答案
#     :return: 返回 BERTScore，范围在[0,1]之间，越大越好
#     """
#     # 计算 BERTScore
#     tokenizer = bert_score.utils.get_tokenizer(model_type="bert-base-multilingual-cased")
#     candidate_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(candidate, add_special_tokens=True)))
#     reference_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(reference, add_special_tokens=True)))
#     max_tokens = 512
#     if len(candidate_tokens) > max_tokens or len(reference_tokens) > max_tokens:
#         # 如果超过最大限制，截断或拆分句子
#         candidate_tokens = candidate_tokens[:max_tokens]
#         reference_tokens = reference_tokens[:max_tokens]
#         # 将 token 转换回字符串
#         candidate = tokenizer.convert_tokens_to_string(candidate_tokens)
#         reference = tokenizer.convert_tokens_to_string(reference_tokens)
#         # 确保截断或拆分后的长度相等
#         min_len = min(len(candidate), len(reference))
#         candidate = candidate[:min_len]
#         reference = reference[:min_len]
#     candidate_answer = [candidate]
#     reference_answer = [reference]
#     P, R, F1 = bert_score.score(candidate_answer, reference_answer, lang="en", verbose=True, model_type="bert-base-multilingual-cased", device='cuda:6', batch_size=1)
#     # print(np.array(F1)[0])
#     return np.array(F1)[0]


def calculate_bert_score(candidates, references):
    """
    计算答案和标准答案之间的 BERTScore
    :param candidate: 候选答案[[],[]...]
    :param reference: 标准答案[[],[]...]
    :return: 返回 BERTScore，范围在[0,1]之间，越大越好
    """
    print("Evaluating bertscore")
    # 计算 BERTScore
    #将所有答案进行检测是否需要进行截断
    max_tokens = 500
    tokenizer = bert_score.utils.get_tokenizer(model_type="bert-base-multilingual-cased")
    # tokenizer.model_max_length = 512
    for i in tqdm(range(len(candidates)), desc='Evaluating bertscore', leave=False):
        candidate = candidates[i]
        reference = references[i]
        # print(candidate," *********** ",reference)
        candidate_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(candidate, add_special_tokens=True)))
        reference_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(reference, add_special_tokens=True)))
        if len(candidate_tokens) > max_tokens or len(reference_tokens) > max_tokens:
            # 如果超过最大限制，截断或拆分句子
            candidate_tokens = candidate_tokens[:max_tokens]
            reference_tokens = reference_tokens[:max_tokens]
            # 将 token 转换回字符串
            candidate = tokenizer.convert_tokens_to_string(candidate_tokens)
            reference = tokenizer.convert_tokens_to_string(reference_tokens)
            # 确保截断或拆分后的长度相等
            min_len = min(len(candidate), len(reference))
            candidates[i] = candidate[:min_len]
            references[i] = reference[:min_len]

    P, R, F1 = bert_score.score(candidates, references, lang="en", verbose=True, model_type="bert-base-multilingual-cased", device='cuda:0', batch_size=16)
    # print(np.array(F1)[0])
    return F1.numpy().tolist()

if __name__ == '__main__':
    # candidate_answer = "A skeptic is someone who doubts or expresses doubt about a claim or idea without being dismissive of it. They are open-minded and approach evidence with an open mind, searching for reasonable explanations and evidence to support their beliefs.\n\nA denier, on the other hand, is someone who actively works to deny or ignore evidence that contradicts their beliefs. They are often characterized by a closed mind and an unwillingness to consider alternative perspectives. They may also use rhetoric or false claims to try to discredit the evidence."
    # reference_answer = "A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason."
    reference_answer = ["A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason.","A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason.","A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason."]
    candidate_answer = ["A skeptic is someone who doubts or expresses doubt about a claim or idea without being dismissive of it. They are open-minded and approach evidence with an open mind, searching for reasonable explanations and evidence to support their beliefs.\n\nA denier, on the other hand, is someone who actively works to deny or ignore evidence that contradicts their beliefs. They are often characterized by a closed mind and an unwillingness to consider alternative perspectives. They may also use rhetoric or false claims to try to discredit the evidence.","Can you explain?\n5. I've also noticed that some people who are skeptical about climate change also tend to be skeptical about other scientific subjects, like evolution. Can you explain that?\n6. What evidence have you seen that supports the theory of evolution?\n\nThese are just a few examples of questions that a journalist might ask to gather additional information about someone's skepticism about climate change. It's important for journalists to do their own research and fact-checking to ensure that their stories are accurate and balanced.","Here are a few definitions that I found online:\nSkeptic: a person who seeks to acquire and validate knowledge by investigation and analysis, especially of a scientific or mathematical nature.\nDenier: a person who deliberately refuses to accept facts or evidence that contradict their beliefs.\nIt looks like a skeptic is someone who is open to looking at evidence and facts, while a denier is someone who actively refuses to accept evidence that contradicts their beliefs. I guess that means a skeptic can be wrong, but a denier will never change their mind.\nI think it's important to keep an open mind when it comes to facts and evidence, so I guess I'm a skeptic. What about you?\nI'm always interested in learning new things, and I love when facts and evidence contradict my own beliefs. That's when I know I'm really learning something!"]
    print(calculate_bert_score(candidate_answer, reference_answer))