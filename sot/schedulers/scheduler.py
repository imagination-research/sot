from abc import ABC, abstractmethod
import argparse
import logging


class Scheduler(ABC):
    def __init__(self, model):
        self._model = model

    def print_info(self):
        logging.info(f"Scheduler: {self.__class__}")

    @staticmethod
    def command_line_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--scheduler_help", action="help")
        return parser

    @classmethod
    def from_command_line_args(cls, args, model):
        args, other_args = cls.command_line_parser().parse_known_args(args)
        return cls(**vars(args), model=model), other_args

    @abstractmethod
    def get_response(self, request):
        pass
