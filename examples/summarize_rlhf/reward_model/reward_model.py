import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        """
        Args:
            model_path: path to the pretrained model
        """
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        """
        Args:
            input_ids: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using :class:`transformers.BertTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.__call__` for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            past_key_values: (`optional`) `torch.FloatTensor` of shape
                `(batch_size, num_layers * num_heads, sequence_length, embed_size_per_head)`:
                Contains pre-computed key and value hidden-states of the attention blocks.
                Can be used to speed up sequential decoding.
                The last two dimensions of the tensor must match the shape of the
                ``sequence_length`` and ``hidden_size`` arguments passed to the ``__call__`` method.
                The number of layers and attention heads may differ between the base model and the checkpoint.
                `What are the key / value states? <../glossary.html#key-value-states>`__
            attention_mask: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                `What are attention masks? <../glossary.html#attention-mask>`__
            token_type_ids: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
                Segment token indices to indicate first and second portions of the inputs.
                Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
                corresponds to a `sentence B` token
                `What are token type IDs? <../glossary.html#token-type-ids>`_
            position_ids: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range ``[0, config.max_position_embeddings - 1]``.
                `What are position IDs? <../glossary.html#position-ids>`_
            head_mask: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
                Mask to nullify selected heads of the self-attention modules.
                Mask values selected in ``[0, 1]``:
                ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
            inputs_embeds: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
                Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            mc_token_ids: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices)``:
                Index of the classification token in each input sequence.
                Selected in the range ``[0, input_ids.size(-1) - 1[``.
            labels: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
                Labels for computing the multiple choice classification loss.
                Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
                of the input tensors. (see `input_ids` above)
            return_dict: (`optional`) ``bool``:
                If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
                plain tuple.
            output_attentions: (`optional`) ``bool``:
                If set to ``True``, the model will return the attentions tensors
                of all attention layers.
                See ``attentions`` under returned tensors for more detail.
            output_hidden_states: (`optional`) ``bool``:
                If set to ``True``, the model will return the hidden states of all layers.
                See ``hidden_states`` under returned tensors for more detail.

        Returns:

        Examples::

            >>> from transformers import BertTokenizer, BertForMultipleChoice
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForMultipleChoice.from_pretrained('bert-base-uncased')

            >>> choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]

            >>> input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True)
                for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
            >>> labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids, labels=labels)

            >>> loss, classification_scores = outputs[:2]

        """
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }
