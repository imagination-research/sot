"""
Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import json
import time

import gradio as gr
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INACTIVE_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
)
from fastchat.model.model_adapter import get_conversation_template

import gradio_web_server
from gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    no_change_text,
    empty_text,
    learn_more_md,
    get_model_description_md,
    ip_expiration_dict,
)
from fastchat.utils import (
    build_logger,
    violates_moderation,
)


logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False


def set_global_vars_named(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named(models, url_params):
    states = (None,) * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8, 4, 2, 1] + [1] * 32)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = (
        gr.Dropdown.update(choices=models, value=model_left, visible=True),
        gr.Dropdown.update(choices=models, value=model_right, visible=True),
    )

    return (
        states
        + selector_updates
        + (gr.Chatbot.update(visible=True),) * num_sides
        + (gr.Textbox.update(visible=True),) * num_sides
        + (
            gr.Textbox.update(visible=True),
            gr.Box.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True),
        )
    )


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (named). ip: {request.client.host}")
    states = [state0, state1]
    for i in range(num_sides):
        states[i].conv.update_last_message(None)
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""] * num_sides
        + [""]
        + [disable_btn] * 2
    )


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {request.client.host}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + [""] * num_sides
        + [""]
        + [disable_btn] * 2
    )


def add_text(
    state0, state1, model_selector0, model_selector1, text, request: gr.Request
):
    ip = request.client.host
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = State(model_selectors[i])

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""] * num_sides
            + [""]
            + [
                no_change_btn,
            ]
            * 2
        )

    if ip_expiration_dict[ip] < time.time():
        logger.info(f"inactive (named). ip: {request.client.host}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""] * num_sides
            + [INACTIVE_MSG]
            + [
                no_change_btn,
            ]
            * 2
        )

    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(
                f"violate moderation (named). ip: {request.client.host}. text: {text}"
            )
            for i in range(num_sides):
                states[i].skip_next = True
            return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + [""] * num_sides
                + [MODERATION_MSG]
                + [
                    no_change_btn,
                ]
                * 2
            )

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {request.client.host}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""] * num_sides
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 2
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""] * num_sides
        + [""]
        + [
            disable_btn,
        ]
        * 2
    )


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (named). ip: {request.client.host}")

    if state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_text,) * num_sides + (no_change_btn,) * 2
        return

    states = [state0, state1]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
            )
        )

    chatbots = [None] * num_sides
    meta_info = [None] * num_sides
    while True:
        stop = True
        for i in range(num_sides):
            try:
                ret = next(gen[i])
                states[i], chatbots[i], meta_info[i] = ret[0], ret[1], ret[2]
                stop = False
            except StopIteration:
                pass

        yield states + chatbots + meta_info + [disable_btn] * 2
        if stop:
            break


def flash_buttons():
    btn_updates = [
        [enable_btn] * 2,
        [enable_btn] * 2,
    ]
    for i in range(10):
        yield btn_updates[i % 2]
        time.sleep(0.2)


def build_side_by_side_ui_named(models):
    notice_markdown = """
### Choose two models to chat with
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides
    meta_info = [None] * num_sides

    # model_description_md = get_model_description_md(models)
    # notice = gr.Markdown(
    #     notice_markdown + model_description_md, elem_id="notice_markdown"
    # )

    with gr.Box(elem_id="share-region-named"):
        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Dropdown(
                        choices=models,
                        value=models[i] if len(models) > i else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        show_label=False, elem_id=f"chatbot", visible=False, height=550
                    )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    meta_info[i] = gr.Textbox(
                        show_label=False,
                        visible=False,
                        container=False,
                        lines=3,
                    )

    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
                container=False,
            )
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)

    with gr.Row() as button_row2:
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(learn_more_md)

    # Register listeners
    btn_list = [
        regenerate_btn,
        clear_btn,
    ]
    regenerate_btn.click(
        regenerate, states, states + chatbots + meta_info + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + meta_info + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(
        clear_history, None, states + chatbots + meta_info + [textbox] + btn_list
    )

    for i in range(num_sides):
        model_selectors[i].change(
            clear_history, None, states + chatbots + meta_info + [textbox] + btn_list
        )

    textbox.submit(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + meta_info + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + meta_info + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    send_btn.click(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + meta_info + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + meta_info + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return (
        states,
        model_selectors,
        chatbots,
        meta_info,
        textbox,
        send_btn,
        button_row2,
        parameter_row,
    )
