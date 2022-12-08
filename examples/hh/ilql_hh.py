import os
import json
import yaml
import trlx
from trlx.data.configs import TRLConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
import torch
import tritonclient.grpc as client_util
from tritonclient.utils import np_to_triton_dtype
import numpy as np

config_path = os.path.join(os.path.dirname(__file__), "configs/ilql_hh.yml")
default_config = yaml.safe_load(open(config_path))
triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "rm-gpt-j")


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    client = client_util.InferenceServerClient(url=triton_host, verbose=False)

    def metric_fn(samples):
        input = tokenizer(samples, padding=True)
        input_ids = np.array(input.input_ids, dtype=np.int32)
        attention_mask = np.array(input.attention_mask, dtype=np.int8)

        inputs = [
            prepare_tensor("input_ids", input_ids),
            prepare_tensor("attention_mask", attention_mask),
        ]

        result = client.infer(triton_model, inputs)
        output_data = result.as_numpy("rewards")
        if output_data is None:
            raise RuntimeError("No output data")
        return {"rewards": output_data[:, -1]}

    fpath = "datasets/rm_labeled_single_context_pairwise.jsonl"
    dataset = [json.loads(line) for line in open(fpath).read().splitlines()]

    prompts, outputs, rewards = [], [], []
    for x in dataset:
        chosen, rejected = x["chosen"], x["rejected"]
        for ix, (a, b) in enumerate(zip(chosen, rejected)):
            if a != b:
                prompts.append(chosen[:ix])
                outputs.append(chosen[ix:])
                rewards.append(x["chosen_reward"])

                samples.append(rejected[:ix])
                outputs.append(rejected[ix:])
                rewards.append(x["rejected_reward"])
                break

    trlx.train(
        dataset=(prompts, outputs, rewards),
        config=config,
        eval_prompts=prompts,
        metric_fn=metric_fn,
    )


if __name__ == "__main__":
    main()