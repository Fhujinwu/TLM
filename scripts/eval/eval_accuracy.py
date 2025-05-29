from eval_utils import extract_logiqa_option, extract_gsm8k_answer_number, extract_math_answer
import json

if __name__ == '__main__':
    paths = [
        "path to your generated_predictions.jsonl",
    ]

    
    ems = []
    for path in paths:
        all_result = []
        labels = []
        with open(path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                # 将每行解析成 JSON 对象
                data = json.loads(line)
                completion = data["predict"]
                result = extract_logiqa_option(completion)  # logiqa dataset
                # result = extract_gsm8k_answer_number(completion)  # gsm8k dataset
                # result = extract_math_answer(completion, data["label"]) # meta_math dataset
                # print(f"位置：{i+1}, {result}")
                # print(f"提取后： {result}--------{i+1}")
                all_result.append(result)
                labels.append(data["label"].strip('\n'))   # .replace(',', '')
                if not result:
                    print(f"位置：{i+1}")
                    print(f"预测文本：{completion}")
            print(all_result[:60])
            print(labels[:60])
            
            compare_res = [a == b for a, b in zip(all_result, labels)]
            print([i+1 for i, _ in enumerate(compare_res) if compare_res[i]==False])
            em = sum(compare_res) / len(all_result)
            ems.append(em)
    
    for path, em in zip(paths, ems):
           
        print(f"{path.split('/')[8]}---{path.split('/')[-4]}---{path.split('/')[-3]}", em)
    