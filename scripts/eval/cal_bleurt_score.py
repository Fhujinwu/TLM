# #计算bleurt指标，值越大越好

from bleurt import score
from tqdm import tqdm
def calculate_bleurt_score(candidate, reference):
    """
    计算答案和标准答案之间的 bleurtScore
    :param candidate: 候选答案
    :param reference: 标准输出
    :return:输出bleurt得分，[0,1]，得分越大越好
    """
    results = []
    checkpoint = "/hujinwu/LLM_Assemble/pretrain_model/bleurt/bleurt-base-128"
    # references = [reference]
    # candidates = [candidate]
    scorer = score.BleurtScorer(checkpoint)
    for i in tqdm(range(len(candidate)), desc='Evaluating bleurtscore', leave=False):
        scores = scorer.score(references=[reference[i]], candidates=[candidate[i]])
        assert isinstance(scores, list) and len(scores) == 1
        results.append(scores[0])
    # print(scores)
    return results

if __name__ == '__main__':
    candidate_answer = "A skeptic is someone who doubts or expresses doubt about a claim or idea without being dismissive of it. They are open-minded and approach evidence with an open mind, searching for reasonable explanations and evidence to support their beliefs.\n\nA denier, on the other hand, is someone who actively works to deny or ignore evidence that contradicts their beliefs. They are often characterized by a closed mind and an unwillingness to consider alternative perspectives. They may also use rhetoric or false claims to try to discredit the evidence."
    reference_answer = "A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason."
    reference_answer = [
        "A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason.",
        "A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason.",
        "A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason."]
    candidate_answer = [
        "A skeptic is someone who doubts or expresses doubt about a claim or idea without being dismissive of it. They are open-minded and approach evidence with an open mind, searching for reasonable explanations and evidence to support their beliefs.\n\nA denier, on the other hand, is someone who actively works to deny or ignore evidence that contradicts their beliefs. They are often characterized by a closed mind and an unwillingness to consider alternative perspectives. They may also use rhetoric or false claims to try to discredit the evidence.",
        "Can you explain?\n5. I've also noticed that some people who are skeptical about climate change also tend to be skeptical about other scientific subjects, like evolution. Can you explain that?\n6. What evidence have you seen that supports the theory of evolution?\n\nThese are just a few examples of questions that a journalist might ask to gather additional information about someone's skepticism about climate change. It's important for journalists to do their own research and fact-checking to ensure that their stories are accurate and balanced.",
        "Here are a few definitions that I found online:\nSkeptic: a person who seeks to acquire and validate knowledge by investigation and analysis, especially of a scientific or mathematical nature.\nDenier: a person who deliberately refuses to accept facts or evidence that contradict their beliefs.\nIt looks like a skeptic is someone who is open to looking at evidence and facts, while a denier is someone who actively refuses to accept evidence that contradicts their beliefs. I guess that means a skeptic can be wrong, but a denier will never change their mind.\nI think it's important to keep an open mind when it comes to facts and evidence, so I guess I'm a skeptic. What about you?\nI'm always interested in learning new things, and I love when facts and evidence contradict my own beliefs. That's when I know I'm really learning something!"]
    print(calculate_bleurt_score(candidate_answer, reference_answer))