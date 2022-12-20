import os
import re
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

    fpath = "datasets/hh-rlhf/helpful-base/test.jsonl"
    dialogues = [json.loads(line) for line in open(fpath).read().splitlines()]
    dialogues = sum([[x["chosen"], x["rejected"]] for x in dialogues], [])
    dialogues = [re.split(r"(\n\nHuman: |\n\nAssistant: )", x)[1:] for x in dialogues]
    rewards = [1, 0] * (len(dialogues) // 2)
    prompts_responses = []
    all_rewards = []
    for xs, r in zip(dialogues, rewards):
        try:
            res = sum(
                [
                    [xs[ix] + xs[ix + 1] + xs[ix + 2], xs[ix + 3]]
                    for ix in range(0, len(xs), 4)
                ],
                [],
            )
            prompts_responses.append(res)
            all_rewards.append(r)
        except IndexError:
            pass

    prompts = ["".join(x[:-1]) for x in prompts_responses[::2]][:64]

    trlx.train(
        dataset=(prompts_responses, all_rewards),
        config=config,
        eval_prompts=prompts,
        metric_fn=metric_fn,
    )


if __name__ == "__main__":
    main()
