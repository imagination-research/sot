from .model import Model


def get_model_class_from_name(name):
    # Lazy import to improve loading speed and reduce libary dependency.
    if name == "openai":
        from .openai_model import OpenAIModel

        return OpenAIModel
    elif name == "fastchat":
        from .fastchat_model import FastChatModel

        return FastChatModel
    elif name == "claude_slack":
        from .claude_slack_model import ClaudeSlackModel

        return ClaudeSlackModel
    else:
        raise ValueError(f"Unknown model name {name}")


__all__ = ["get_model_class_from_name", "Model"]
