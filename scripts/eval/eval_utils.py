import re
import sys

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False




def extract_gsm8k_answer_number(completion):

    # 1. 按照空格和换行符分割字符串
    tokens = re.split(r'[\s\n\n]+', completion)
    # 2. 过滤掉不包含数字的单词(支持负数)
    tokens_with_numbers = [token for token in tokens if re.search(r'-?\d', token)]
    # 3. 去除与数字无关的符号（保留逗号和小数点）
    cleaned_numbers = [re.sub(r'[^\d,\.-]', '', token) for token in tokens_with_numbers]
    
    if cleaned_numbers:
        extracted_number = cleaned_numbers[-1].replace(',', '').strip('.')
        if extracted_number.count('.') > 1 or extracted_number.count('-') > 1 or not re.match(r'^-?\d*\.?\d*$', extracted_number):   # 小数点大于1，不符合规范
            return None
        try:
            return str(round(float(extracted_number))) 
        except:
            print(extracted_number)
            return None

    return  None



def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = sys.maxsize
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def process_results(doc, completion, answer, invalid_outputs):
    split_ans = completion.split('The answer is: ') if len(completion.split('The answer is: '))>1 else completion.split('The answer is ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        print(f"extract_ans: {extract_ans}")
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        # pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2



def extract_logiqa_option(completion: str):
    """
    答案的形式可能有：
    ["Answer: D.", "Answer: , "The correct answer is: C.", "D. xxx Explanation", "The correct answer is D"]
    """
    result = ""
    if re.match(r"^[A-D]\.\s", completion):
        result = completion[0]
    elif all(keyword in completion for keyword in ["Answer: ", "Explanation"]):  # 先找到格式占比最大的，即出现"Answer:"
        # completion.split("Answer:")[1].strip()
        match = re.search(r'Answer:\s*(.*?)\s*Explanation', completion, re.DOTALL)  # 在字符串中查找第一个符合正则表达式的匹配
        if match:
            tmp = match.group(1)
        else:
            print(f'他妈哒: {completion}')
            tmp = re.split(r'Answer: ', completion)[1]
        match = re.search(r'\b[A-D]\b', tmp)
        if match: 
            result = match.group()
            # print(result)
    elif 'Explanation' in completion:
        tmp = re.split(r'Explanation', completion)[0]
        match = re.search(r'\b[A-D]\b', tmp)
        if match:
            result = match.group()
        else:
            print('wrong-0')
    elif any(keyword in completion for keyword in ["Answer: ", "answer is"]):
        tmp = re.split(r"Answer:\s*|answer is\s*", completion, maxsplit=1)[1]
        match = re.search(r'\b[A-D]\b', tmp)
        if match:
            result = match.group()
        else:
            print('wrong-1')
    else:
        match = re.search(r'\b[A-D]\b', completion)
        if match:
            result = match.group()
        else: 
            print('wrong-2')
    return result

def extract_math_answer(completion: str, label: str):  # 可能要根据正确答案反向匹配预测答案
    # 先考虑特殊情况，label中出现\\text{xx}的
    if any(key in label for key in ["\\text{", ":"]):
        if "\\text{" in label:
            content = re.search(r'\\text{(.*?)}', label).group(1)
        elif ":" in label:
            content = label
        temp_completion = re.split(r"[Tt]herefore|[Ss]o", completion)
        if len(temp_completion) > 1:
            if content in temp_completion[-1]:   # 正确答案在预测结果中出现，则返回正确答案，避免格式不一样如\\text和\\textbf
                return label
        return None
    elif label.startswith('('):
        print(f"以左括号开头: {label}")
        content = re.search(r"\((.*?)", label).group(1)
        temp_completion = re.split(r"[Tt]herefore|[Ss]o", completion)
        if len(temp_completion) > 1:
            if content in temp_completion[-1].replace(" ", ""):   # 正确答案在预测结果中出现，则返回正确答案，避免格式不一样如\\text和\\textbf
                return label
        return None


    # 首先找到 \boxed{}，提取包围在大括号里的内容，一定是预测答案
    match = re.search(r'\\boxed\{(.*)\}', completion)  # 注意这里要贪婪匹配，例如出现Therefore, the value of $a + b$ is $\\boxed{\\frac{7}{2}}$.
    if match:
        result = match.group(1)
        result = post_process(result)
        # print(f"第一步提取出答案：{result}")
    elif matches := re.findall(r'\\frac\{\d\}\{\d\}', completion):
        result = matches[-1]
        result = post_process(result)
        # print(f"第二步提取出答案：{result}")
    else:
        result = extract_gsm8k_answer_number(completion)
    
    return result

def post_process(string: str):
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # remove space
    string = string.replace(" ", "")

    return string



if __name__ == '__main__':
    pass