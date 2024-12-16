# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..utils import DummyObject, requires_backends


class KolorsImg2ImgPipeline(metaclass=DummyObject):
    _backends = ["torch", "transformers", "sentencepiece"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "transformers", "sentencepiece"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "transformers", "sentencepiece"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "transformers", "sentencepiece"])


class KolorsPAGPipeline(metaclass=DummyObject):
    _backends = ["torch", "transformers", "sentencepiece"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "transformers", "sentencepiece"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "transformers", "sentencepiece"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "transformers", "sentencepiece"])


class KolorsPipeline(metaclass=DummyObject):
    _backends = ["torch", "transformers", "sentencepiece"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "transformers", "sentencepiece"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "transformers", "sentencepiece"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "transformers", "sentencepiece"])
