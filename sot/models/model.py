from abc import ABC, abstractmethod
import argparse


class Model(ABC):
    @staticmethod
    def command_line_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_help", action="help")
        return parser

    @classmethod
    def from_command_line_args(cls, args):
        args, other_args = cls.command_line_parser().parse_known_args(args)
        return cls(**vars(args)), other_args

    @abstractmethod
    def get_response(self, requests):
        pass
