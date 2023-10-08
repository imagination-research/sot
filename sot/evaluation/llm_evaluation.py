import logging
import re
import numpy as np


def is_float(str):
    try:
        float(str)
        return True
    except Exception:
        return False


def fastchat_evaluation(model, template, question, answer_1, answer_2, prompt):
    """
    Based on https://github.com/lm-sys/FastChat/tree/main/fastchat/eval
    """
    request = template.format(
        question=question, answer_1=answer_1, answer_2=answer_2, prompt=prompt
    )
    try:
        response = model.get_response([request])[0]["text"]
    except Exception as e:
        logging.error(f"Exception: {e}")
        response = "-2,-2"
    score_pair = response.split("\n")[0].replace(",", " ").split(" ")
    if len(score_pair) == 2 and is_float(score_pair[0]) and is_float(score_pair[1]):
        score_pair = list(map(float, score_pair))
    else:
        score_pair = re.findall(r"\(([\d]+\.*[\d]*),\ *([\d]+\.*[\d]*)\)", response)
        if len(score_pair) != 1:
            logging.error(f"Invalid score pair: {score_pair}")
            score_pair = [-1, -1]
        else:
            score_pair = list(map(float, score_pair[0]))
    return {
        "score_pair": score_pair,
        "response": response,
        "evaluation_request": request,
    }


def llm_zoo_evaluation(model, question, answers, prompt):
    """
    Based on https://github.com/FreedomIntelligence/LLMZoo/tree/main/llmzoo/eval
    """
    NUM2STR = {
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        20: "twenty",
    }
    num_answers = len(answers)
    num_answers_str = NUM2STR[num_answers]

    prompt = prompt.format(num=num_answers, num_str=num_answers_str)
    request = f"[Question]\n{question}\n\n"
    request += "\n\n".join(
        [
            f"[Assistant {i}]\n{ans}\n\n[End of Assistant {i}]"
            for i, ans in enumerate(answers, start=1)
        ]
    )
    request += "\n\n" + f"[System]\n{prompt}\n\n"

    try:
        response = model.get_response([request])[0]["text"]
        order = _parse_llm_zoo_order_cot(response, num_answers)
    except Exception as e:
        logging.error(f"Exception: {e}")
        response = "-2,-2"
        order = [-2, -2]

    return {
        "order": order,
        "response": response,
        "evaluation_request": request,
    }


def _parse_llm_zoo_order_cot(review, n_ans):
    """
    From:
    https://github.com/FreedomIntelligence/LLMZoo/blob/main/llmzoo/eval/eval_gpt_review_all.py
    """
    review = re.sub(r">=", ">", review)
    review = re.sub(r">>", ">", review)
    try:
        ls = re.findall(r"(Assistant \d+( [>=] Assistant \d+)+)", review.strip())
        order_texts = [x[0] for x in ls]
        idxs = np.where(
            np.array(
                [len(re.findall(r"Assistant", text)) == n_ans for text in order_texts]
            )
        )[0]
        if idxs.shape[0] == 0:
            return [-1] * n_ans
        order_text = order_texts[idxs[0]]

        ordered_assist = [int(x) for x in re.findall(r"\d+", order_text)]
        ordered_comp = re.findall(r"[>=]", order_text)

        order = [0] * n_ans
        cur_order = 1
        num_eq = 0
        order[ordered_assist[0] - 1] = cur_order
        for comp, assist in zip(ordered_comp, ordered_assist[1:]):
            if comp == ">":
                cur_order += num_eq + 1
                order[assist - 1] = cur_order
                num_eq = 0
            else:
                order[assist - 1] = cur_order
                num_eq += 1
        return order

    except Exception:
        # print(e)
        # print('error', review)
        return [-1] * n_ans
