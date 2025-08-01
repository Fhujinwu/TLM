import json
from cal_bert_score import calculate_bert_score
from cal_bleurt_score import calculate_bleurt_score
import numpy as np
from rouge_score import rouge_scorer
from nltk import sent_tokenize
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
#读取文件
def read(file_path):
    """
    :param path: 其他测试文件夹的路径
    :return:输出一个数组data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 从每一行中加载 JSON 对象
            json_object = json.loads(line)
            data.append(json_object)
    return data

def calculate(datas):
    #读取所有候选答案candidate，和标准输出reference
    candidate = []
    reference = []
    for data in datas:
        candidate.append(data['predict'])
        reference.append(data['label'])
    #计算bertscore
    bertscore = calculate_bert_score(candidate, reference)
    # print(bertscore)
    # print(bartscore)
    # 计算bleurt
    bleurtscore = calculate_bleurt_score(candidate, reference)
    # print(bleurtscore)
    bertscore = np.array(bertscore)
    bleurtscore = np.array(bleurtscore)
    print("bertscore:", np.mean(bertscore), "bleurtscore:", np.mean(bleurtscore))
    return np.mean(bertscore), np.mean(bleurtscore)


def pre_rouge_processing(summary):
    summary = summary.replace("<n>", " ")
    summary = "\n".join(sent_tokenize(summary))
    return summary

def calue_rougscore(datas):
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    rouge1 = 0
    rouge2 = 0
    rougeL = 0
    rougeLsum = 0
    candidate = []
    reference = []
    for data in datas:
        candidate.append(data['predict'])
        reference.append(data['label'])
    length = len(candidate)
    rougeLsum_ls = []
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True, split_summaries=True)
    for i in tqdm(range(len(candidate)), desc='Evaluating rouge score', leave=False):
        scores = scorer.score(reference[i], pre_rouge_processing(candidate[i]))
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
        rougeLsum += scores['rougeLsum'].fmeasure
        rougeLsum_ls.append(scores['rougeLsum'].fmeasure)

    return rouge1/length, rouge2/length, rougeL/length, rougeLsum/length


#计算BLEU 得分
def calculate_bleu_score(datas):
    candidate = []
    reference = []
    for data in datas:
        candidate.append(data['predict'])
        reference.append(data['label'])
    bleu_sum = 0
    for i in range(len(candidate)):
        #将句子进行分词
        ref = [nltk.tokenize.word_tokenize(reference[i])]
        can = nltk.tokenize.word_tokenize(candidate[i])
        #计算BLEU得分
        smooth = SmoothingFunction().method4
        score = sentence_bleu(ref, can, smoothing_function=smooth)
        bleu_sum += score
    return bleu_sum/len(candidate)

if __name__ == '__main__':

    
    datas_paths = [
        # 'path to your generated_predictions.jsonl',
    ]

    results = []
    for datas_path in datas_paths:
        datas = read(datas_path)[:]
        rouge1, rouge2, rougeL, rougeLsum = calue_rougscore(datas)
        bertscroe, bleurtscore = calculate(datas)
        bleu = calculate_bleu_score(datas)
        # result = (rouge1, rouge2, rougeL, rougeLsum)
        result = (bertscroe, bleurtscore, bleu, rouge1, rouge2, rougeL, rougeLsum)
        results.append(result)
        # print(f"bertscore: {bertscroe}, bleurtscore: {bleurtscore}, bleu: {bleu}, rouge1: {rouge1}, rouge2: {rouge2}, rougeL: {rougeL}, rougeLsum: {rougeLsum}")
    
    for path, result in zip(datas_paths, results):
        print(f"Path: {path}")
        print(f"bertscore: {result[0]}, bleurtscore: {result[1]}, bleu: {result[2]}, rouge1: {result[3]}, rouge2: {result[4]}, rougeL: {result[5]}, rougeLsum: {result[6]}")