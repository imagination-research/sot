"""
The gradio demo server with multiple tabs.
It supports chatting with a single model or chatting with two models side-by-side.
"""

import argparse
import pickle
import time

import gradio as gr

from fastchat.constants import (
    SESSION_EXPIRATION_TIME,
)

import gradio_block_sot_named
import gradio_web_server

from gradio_block_sot_named import (
    build_side_by_side_ui_named,
    load_demo_side_by_side_named,
    set_global_vars_named,
)
from gradio_web_server import (
    set_global_vars,
    block_css,
    get_model_list,
    ip_expiration_dict,
)

# from fastchat.serve.monitor.monitor import build_leaderboard_tab
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    parse_gradio_auth_creds,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")


def load_demo(url_params, request: gr.Request):
    global models

    ip = request.client.host
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    if args.model_list_mode == "reload":
        models = get_model_list(
            args.controller_url, args.add_chatgpt, args.add_claude, args.add_palm
        )

    side_by_side_named_updates = load_demo_side_by_side_named(models, url_params)
    return side_by_side_named_updates


def build_demo(models):
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Chatbot", id=2):
                (
                    c_states,
                    c_model_selectors,
                    c_chatbots,
                    c_meta_info,
                    c_textbox,
                    c_send_btn,
                    c_button_row2,
                    c_parameter_row,
                ) = build_side_by_side_ui_named(models)
                c_list = (
                    c_states
                    + c_model_selectors
                    + c_chatbots
                    + c_meta_info
                    + [
                        c_textbox,
                        c_send_btn,
                        c_button_row2,
                        c_parameter_row,
                    ]
                )

        url_params = gr.JSON(visible=False)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")
        demo.load(
            load_demo,
            [url_params],
            c_list,
            _js=get_window_url_params_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link.",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://0.0.0.0:21001",
        help="The address of the controller.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue.",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time.",
    )
    parser.add_argument(
        "--moderate", action="store_true", help="Enable content moderation"
    )
    parser.add_argument(
        "--add-chatgpt",
        action="store_true",
        help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        help="Add Anthropic's Claude models (claude-v1, claude-instant-v1)",
    )
    parser.add_argument(
        "--add-palm",
        action="store_true",
        help="Add Google's PaLM model (PaLM 2 for Chat: chat-bison@001)",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help=(
            "Set the gradio authentication file path. The file should contain one or"
            ' more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"'
        ),
        default=None,
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate)
    set_global_vars_named(args.moderate)
    models = get_model_list(
        args.controller_url, False, args.add_chatgpt, args.add_claude, args.add_palm
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
    )
