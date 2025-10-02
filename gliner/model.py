import json
import os
import re
import warnings
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Union

import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file

from .config import GLiNERConfig
from .data_processing import SpanProcessor, SpanBiEncoderProcessor, TokenProcessor, TokenBiEncoderProcessor
from .data_processing.collator import DataCollator, DataCollatorWithPadding
from .data_processing.tokenizer import WordsSplitter
from .decoding import SpanDecoder, TokenDecoder
from .decoding.trie import LabelsTrie
from .evaluation import Evaluator
from .modeling.base import BaseModel, SpanModel, TokenModel
from .infer_packing import InferencePackingConfig
from .onnx.model import BaseORTModel, SpanORTModel, TokenORTModel


class GLiNER(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        config: GLiNERConfig,
        model: Optional[Union[BaseModel, BaseORTModel]] = None,
        tokenizer: Optional[Union[str, AutoTokenizer]] = None,
        words_splitter: Optional[Union[str, WordsSplitter]] = None,
        data_processor: Optional[Union[SpanProcessor, TokenProcessor]] = None,
        encoder_from_pretrained: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the GLiNER model.

        Args:
            config (GLiNERConfig): Configuration object for the GLiNER model.
            model (Optional[Union[BaseModel, BaseORTModel]]): GLiNER model to use for predictions. Defaults to None.
            tokenizer (Optional[Union[str, AutoTokenizer]]): Tokenizer to use. Can be a string (path or name) or an AutoTokenizer instance. Defaults to None.
            words_splitter (Optional[Union[str, WordsSplitter]]): Words splitter to use. Can be a string or a WordsSplitter instance. Defaults to None.
            data_processor (Optional[Union[SpanProcessor, TokenProcessor]]): Data processor - object that prepare input to a model. Defaults to None.
            encoder_from_pretrained (bool): Whether to load the encoder from a pre-trained model or init from scratch. Defaults to True.
        """
        super().__init__()
        self.config = config

        if tokenizer is None and data_processor is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)

        if words_splitter is None and data_processor is None:
            words_splitter = WordsSplitter(config.words_splitter_type)

        if config.span_mode == "token_level":
            if model is None:
                self.model = TokenModel(config, encoder_from_pretrained, cache_dir=cache_dir)
            else:
                self.model = model
            if data_processor is None:
                if config.labels_encoder is not None:
                    labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder, cache_dir=cache_dir)
                    self.data_processor = TokenBiEncoderProcessor(config, tokenizer, words_splitter, labels_tokenizer)
                else:
                    self.data_processor = TokenProcessor(config, tokenizer, words_splitter)

            else:
                self.data_processor = data_processor
            self.decoder = TokenDecoder(config)
        else:
            if model is None:
                self.model = SpanModel(config, encoder_from_pretrained, cache_dir=cache_dir)
            else:
                self.model = model
            if data_processor is None:
                if config.labels_encoder is not None:
                    labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder, cache_dir=cache_dir)
                    self.data_processor = SpanBiEncoderProcessor(config, tokenizer, words_splitter, labels_tokenizer)
                else:
                    if config.labels_decoder is not None:
                        decoder_tokenizer = AutoTokenizer.from_pretrained(config.labels_decoder, cache_dir=cache_dir, add_prefix_space=True)
                        if decoder_tokenizer.pad_token is None:
                            decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
                    else:
                        decoder_tokenizer = None
                    self.data_processor = SpanProcessor(config, tokenizer, words_splitter, decoder_tokenizer=decoder_tokenizer)
            else:
                self.data_processor = data_processor
            self.decoder = SpanDecoder(config)

        self.model.data_processor = self.data_processor # for debugging purposes
        
        if config.vocab_size != -1 and config.vocab_size != len(
            self.data_processor.transformer_tokenizer
        ):
            warnings.warn(f"""Vocab size of the model ({config.vocab_size}) does't match length of tokenizer ({len(self.data_processor.transformer_tokenizer)}). 
                            You should to consider manually add new tokens to tokenizer or to load tokenizer with added tokens.""")

        if isinstance(self.model, BaseORTModel):
            self.onnx_model = True
        else:
            self.onnx_model = False

        # to suppress an AttributeError when training
        self._keys_to_ignore_on_save = None
        self._inference_packing_config: Optional[InferencePackingConfig] = None

    def forward(self, *args, **kwargs):
        """Wrapper function for the model's forward pass."""
        output = self.model(*args, **kwargs)
        return output

    @property
    def device(self):
        if self.onnx_model:
            providers = self.model.session.get_providers()
            if 'CUDAExecutionProvider' in providers:
                return torch.device('cuda')
            return torch.device('cpu')
        device = next(self.model.parameters()).device
        return device

    def resize_token_embeddings(
        self,
        add_tokens,
        set_class_token_index=True,
        add_tokens_to_tokenizer=True,
        pad_to_multiple_of=None,
    ) -> nn.Embedding:
        """
        Resize the token embeddings of the model.

        Args:
            add_tokens: The tokens to add to the embedding layer.
            set_class_token_index (bool, optional): Whether to set the class token index. Defaults to True.
            add_tokens_to_tokenizer (bool, optional): Whether to add the tokens to the tokenizer. Defaults to True.
            pad_to_multiple_of (int, optional): If set, pads the embedding size to be a multiple of this value. Defaults to None.

        Returns:
            nn.Embedding: The resized embedding layer.
        """
        if set_class_token_index:
            self.config.class_token_index = (
                len(self.data_processor.transformer_tokenizer) + 1
            )
        if add_tokens_to_tokenizer:
            self.data_processor.transformer_tokenizer.add_tokens(add_tokens)
        new_num_tokens = len(self.data_processor.transformer_tokenizer)
        model_embeds = self.model.token_rep_layer.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.vocab_size = model_embeds.num_embeddings

        if self.config.encoder_config is not None:
            self.config.encoder_config.vocab_size = model_embeds.num_embeddings

        return model_embeds

    def configure_inference_packing(
        self, config: Optional[InferencePackingConfig]
    ) -> None:
        """Configure default packing behaviour for inference calls.

        Passing ``None`` disables packing by default. Individual inference
        methods accept a ``packing_config`` argument to override this setting
        on a per-call basis.
        """

        self._inference_packing_config = config

    def prepare_texts(self, texts: List[str]):
        """
        Prepare inputs for the model.

        Args:
            texts (str): The input text or texts to process.
            labels (str): The corresponding labels for the input texts.
        """
        all_tokens = []
        all_start_token_idx_to_text_idx = []
        all_end_token_idx_to_text_idx = []

        for text in texts:
            tokens = []
            start_token_idx_to_text_idx = []
            end_token_idx_to_text_idx = []
            for token, start, end in self.data_processor.words_splitter(text):
                tokens.append(token)
                start_token_idx_to_text_idx.append(start)
                end_token_idx_to_text_idx.append(end)
            all_tokens.append(tokens)
            all_start_token_idx_to_text_idx.append(start_token_idx_to_text_idx)
            all_end_token_idx_to_text_idx.append(end_token_idx_to_text_idx)

        input_x = [{"tokenized_text": tk, "ner": None} for tk in all_tokens]

        return input_x, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx


    def prepare_model_inputs(self, texts: List[str], labels: List[str], prepare_entities: bool = True):
        """
        Prepare inputs for the model.

        Args:
            texts (str): The input text or texts to process.
            labels (str): The corresponding labels for the input texts.
        """
        # preserving the order of labels
        labels = list(dict.fromkeys(labels))

        class_to_ids = {k: v for v, k in enumerate(labels, start=1)}
        id_to_classes = {k: v for v, k in class_to_ids.items()}

        input_x, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx = self.prepare_texts(texts)

        raw_batch = self.data_processor.collate_raw_batch(input_x, labels,
                                                        class_to_ids = class_to_ids,
                                                        id_to_classes = id_to_classes)
        raw_batch["all_start_token_idx_to_text_idx"] = all_start_token_idx_to_text_idx
        raw_batch["all_end_token_idx_to_text_idx"] = all_end_token_idx_to_text_idx

        model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=False,
                                                        prepare_entities=prepare_entities)
        model_input.update(
            {
                "span_idx": raw_batch["span_idx"] if "span_idx" in raw_batch else None,
                "span_mask": raw_batch["span_mask"]
                if "span_mask" in raw_batch
                else None,
                "text_lengths": raw_batch["seq_length"],
            }
        )

        device = self.device
        for key in model_input:
            if model_input[key] is not None and isinstance(
                model_input[key], torch.Tensor
            ):
                model_input[key] = model_input[key].to(device)

        return model_input, raw_batch

    def predict_entities(
        self, text, labels, flat_ner=True, threshold=0.5, multi_label=False, **kwargs
    ):
        """
        Predict entities for a single text input.

        Args:
            text: The input text to predict entities for.
            labels: The labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per entity. Defaults to False.

        Returns:
            The list of entity predictions.
        """
        return self.run(
            [text],
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            **kwargs
        )[0]

    def batch_predict_entities(
        self, texts, labels, flat_ner=True, threshold=0.5, multi_label=False, **kwargs
    ):
        """
        DEPRECATED: Use `run` instead.

        This method will be removed in a future release. It now forwards to
        `GLiNER.run(...)` to perform inference.

        Args:
            texts (List[str]): Input texts.
            labels (List[str]): Labels to predict.
            flat_ner (bool, optional): Use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold. Defaults to 0.5.
            multi_label (bool, optional): Allow multiple labels per token/entity. Defaults to False.
            **kwargs: Extra arguments forwarded to `run` (e.g., batch_size).
        """
        warnings.warn(
            "GLiNER.batch_predict_entities is deprecated and will be removed in a future release. "
            "Please use GLiNER.run instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.run(
            texts,
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            **kwargs,
        )

    def set_labels_trie(self, labels: List[str]):
        """
        Initializing the labels trie

        Args:
            labels (List[str]): Labels that will be used.
        """ 
        tokenized_labels = []
        if self.data_processor.decoder_tokenizer is None:
            raise NotImplementedError("Label trie is implemented only to models with decoder.")
        for label in labels:
            tokens = self.data_processor.decoder_tokenizer.encode(label) # type: ignore
            if tokens[0] == self.data_processor.decoder_tokenizer.bos_token_id:
                tokens = tokens[1:]
            tokens.append(self.data_processor.decoder_tokenizer.eos_token_id)
            tokenized_labels.append(tokens) # type: ignore
        trie = LabelsTrie(tokenized_labels)
        return trie
    
    def generate_labels(self, model_output, **gen_kwargs):
        """
        Generate (or rewrite) the textual class labels for each example in the batch.

        Parameters
        ----------
        model_output : Namespace
            Must expose `decoder_embedding` (FloatTensor, shape [N, L, H]) and
            `decoder_embedding_mask` (BoolTensor, shape [N, L]), where N is the total
            number of label placeholders across the whole minibatch.
        **gen_kwargs
            Extra args forwarded to `self.model.generate_labels` (e.g. `max_new_tokens`).

        Returns
        -------
        list[dict[int, str]]
            One dict per example, mapping the *new* 1-based label-ids to the
            generated label strings.
        """
        # Unpack
        dec_embeds = model_output.decoder_embedding            # [N, L, H]
        if dec_embeds is None:
            return []
        
        dec_mask = model_output.decoder_embedding_mask       # [N, L]

        gen_ids = self.model.generate_labels(dec_embeds, dec_mask, 
                                                max_new_tokens=gen_kwargs.pop("max_new_tokens", 15),
                                                eos_token_id=self.data_processor.decoder_tokenizer.eos_token_id,
                                                do_sample = gen_kwargs.pop("do_sample", True),
                                                temperature = gen_kwargs.pop("temperature", 0.01),
                                                no_repeat_ngram_size = gen_kwargs.pop("no_repeat_ngram_size", 1),
                                                repetition_penalty = gen_kwargs.pop("repetition_penalty", 1.1),
                                                **gen_kwargs)  # [N, S]

        gen_texts = self.data_processor.decoder_tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True
        ) 
        return gen_texts
    
    @torch.no_grad()
    def run(
        self,
        texts,
        labels,
        flat_ner=True,
        threshold=0.5,
        multi_label=False,
        batch_size=8,
        gen_constraints=None,
        num_gen_sequences=1,
        packing_config: Optional[InferencePackingConfig] = None,
        **gen_kwargs,
    ):
        """
        Predict entities for a batch of texts.

        Args:
            texts (List[str]): A list of input texts to predict entities for.
            labels (List[str]): A list of labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per token. Defaults to False.
            packing_config (Optional[InferencePackingConfig], optional):
                Configuration describing how to pack encoder inputs. When ``None``
                the instance-level configuration set via
                :meth:`configure_inference_packing` is used.

        Returns:
            The list of lists with predicted entities.
        """
        self.eval()
        # raw input preparation
        if isinstance(texts, str):
            texts = [texts]
        input_x, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx = self.prepare_texts(texts)
        
        collator = DataCollator(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
            entity_types=labels,
        )
        data_loader = torch.utils.data.DataLoader(
            input_x, batch_size=batch_size, shuffle=False, collate_fn=collator
        )

        outputs = []
        active_packing = (
            packing_config
            if packing_config is not None
            else self._inference_packing_config
        )
        # Iterate over data batches
        for batch in data_loader:
            # Move the batch to the appropriate device
            if not self.onnx_model:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

            # Perform predictions
            if active_packing is not None and not self.onnx_model:
                model_inputs = dict(batch)
                model_inputs["packing_config"] = active_packing
            else:
                model_inputs = batch

            model_output = self.model(**model_inputs, threshold=threshold)
            model_logits = model_output[0]

            if not isinstance(model_logits, torch.Tensor):
                model_logits = torch.from_numpy(model_logits)
            
            gen_labels = None
            id_to_classes = batch["id_to_classes"]
            if self.config.labels_decoder is not None:
                if gen_constraints is not None:
                    labels_trie = self.set_labels_trie(gen_constraints)
                else:
                    labels_trie = None
                gen_labels = self.generate_labels(model_output, labels_trie=labels_trie, 
                                                  num_return_sequences=num_gen_sequences, **gen_kwargs)

            decoded_outputs = self.decoder.decode(
                batch["tokens"],
                id_to_classes,
                model_logits,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
                gen_labels=gen_labels,
                sel_idx = model_output.decoder_span_idx,
                num_gen_sequences=num_gen_sequences
            )
            outputs.extend(decoded_outputs)

        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, gen_ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                ent_details =  {
                        "start": start_token_idx_to_text_idx[start_token_idx],
                        "end": end_token_idx_to_text_idx[end_token_idx],
                        "text": texts[i][start_text_idx:end_text_idx],
                        "label": ent_type,
                        "score": ent_score,
                    }
                if gen_ent_type is not None:
                    ent_details['generated labels'] = gen_ent_type
                entities.append(ent_details)

            all_entities.append(entities)

        return all_entities

    def predict_with_embeds(
        self, text, labels_embeddings, labels, flat_ner=True, threshold=0.5, multi_label=False
    ):
        """
        Predict entities for a single text input.

        Args:
            text: The input text to predict entities for.
            labels: The labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per entity. Defaults to False.

        Returns:
            The list of entity predictions.
        """
        return self.batch_predict_with_embeds(
            [text],
            labels_embeddings,
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )[0]

    @torch.no_grad()
    def batch_predict_with_embeds(
        self,
        texts,
        labels_embeddings,
        labels,
        flat_ner=True,
        threshold=0.5,
        multi_label=False,
        packing_config: Optional[InferencePackingConfig] = None,
    ):
        """
        Predict entities for a batch of texts.

        Args:
            texts (List[str]): A list of input texts to predict entities for.
            labels (List[str]): A list of labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per token. Defaults to False.

        Returns:
            The list of lists with predicted entities.
        """

        model_input, raw_batch = self.prepare_model_inputs(texts, labels, prepare_entities = False)

        active_packing = (
            packing_config
            if packing_config is not None
            else self._inference_packing_config
        )

        model_kwargs = dict(model_input)
        if active_packing is not None and not self.onnx_model:
            model_kwargs["packing_config"] = active_packing

        model_output = self.model(labels_embeddings = labels_embeddings, **model_kwargs)[0]

        if not isinstance(model_output, torch.Tensor):
            model_output = torch.from_numpy(model_output)

        outputs = self.decoder.decode(
            raw_batch["tokens"],
            raw_batch["id_to_classes"],
            model_output,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )

        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = raw_batch["all_start_token_idx_to_text_idx"][
                i
            ]
            end_token_idx_to_text_idx = raw_batch["all_end_token_idx_to_text_idx"][i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                entities.append(
                    {
                        "start": start_token_idx_to_text_idx[start_token_idx],
                        "end": end_token_idx_to_text_idx[end_token_idx],
                        "text": texts[i][start_text_idx:end_text_idx],
                        "label": ent_type,
                        "score": ent_score,
                    }
                )
            all_entities.append(entities)

        return all_entities

    def evaluate(
        self,
        test_data,
        flat_ner=False,
        multi_label=False,
        threshold=0.5,
        batch_size=12,
        entity_types=None,
    ):
        """
        Evaluate the model on a given test dataset.

        Args:
            test_data (List[Dict]): The test data containing text and entity annotations.
            flat_ner (bool): Whether to use flat NER. Defaults to False.
            multi_label (bool): Whether to use multi-label classification. Defaults to False.
            threshold (float): The threshold for predictions. Defaults to 0.5.
            batch_size (int): The batch size for evaluation. Defaults to 12.
            entity_types (Optional[List[str]]): List of entity types to consider. Defaults to None.

        Returns:
            tuple: A tuple containing the evaluation output and the F1 score.
        """
        self.eval()
        # Create the dataset and data loader
        dataset = test_data
        collator = DataCollator(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
            entity_types=entity_types,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
        )

        all_preds = []
        all_trues = []

        # Iterate over data batches
        for batch in data_loader:
            # Move the batch to the appropriate device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Perform predictions
            model_output = self.model(**batch)[0]

            if not isinstance(model_output, torch.Tensor):
                model_output = torch.from_numpy(model_output)

            decoded_outputs = self.decoder.decode(
                batch["tokens"],
                batch["id_to_classes"],
                model_output,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
            )
            all_preds.extend(decoded_outputs)
            all_trues.extend(batch["entities"])
        # Evaluate the predictions
        evaluator = Evaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()

        return out, f1

    def encode_labels(self, labels: List[str], batch_size: int = 8) -> torch.FloatTensor:
        """
        Embedding of labels.

        Args:
            labels (List[str]): A list of labels.
            batch_size (int): Batch size for processing labels.

        Returns:
            labels_embeddings (torch.FloatTensor): Tensor containing label embeddings.
        """
        if self.config.labels_encoder is None:
            raise NotImplementedError("Labels pre-encoding is supported only for bi-encoder model.")

        # Create a DataLoader for efficient batching
        dataloader = DataLoader(labels, batch_size=batch_size, collate_fn=lambda x: x)

        labels_embeddings = []

        for batch in tqdm(dataloader, desc="Encoding labels"):
            tokenized_labels = self.data_processor.labels_tokenizer(batch, return_tensors='pt',
                                                                truncation=True, padding="max_length").to(self.device)
            with torch.no_grad():  # Disable gradient calculation for inference
                curr_labels_embeddings = self.model.token_rep_layer.encode_labels(**tokenized_labels)
            labels_embeddings.append(curr_labels_embeddings)

        return torch.cat(labels_embeddings, dim=0)

    def predict(self, batch, flat_ner=False, threshold=0.5, multi_label=False):
        """
        Predict the entities for a given batch of data.

        Args:
            batch (Dict): A batch of data containing tokenized text and other relevant fields.
            flat_ner (bool): Whether to use flat NER. Defaults to False.
            threshold (float): The threshold for predictions. Defaults to 0.5.
            multi_label (bool): Whether to use multi-label classification. Defaults to False.

        Returns:
            List: Predicted entities for each example in the batch.
        """
        model_output = self.model(**batch)[0]

        if not isinstance(model_output, torch.Tensor):
            model_output = torch.from_numpy(model_output)

        decoded_outputs = self.decoder.decode(
            batch["tokens"],
            batch["id_to_classes"],
            model_output,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )

        return decoded_outputs

    def compile(self):
        self.model = torch.compile(self.model)

    def compile_for_training(self):
        print("Compiling transformer encoder...")
        self.model.token_rep_layer = torch.compile(self.model.token_rep_layer)
        print("Compiling RNN...")
        self.model.rnn = torch.compile(self.model.rnn)
        if hasattr(self.model, "span_rep_layer"):
            print("Compiling span representation layer...")
            self.model.span_rep_layer = torch.compile(self.model.span_rep_layer)
        if hasattr(self.model, "prompt_rep_layer"):
            print("Compiling prompt representation layer...")
            self.model.prompt_rep_layer = torch.compile(self.model.prompt_rep_layer)
        if hasattr(self.model, "scorer"):
            print("Compiling scorer...")
            self.model.scorer = torch.compile(self.model.scorer)

    def set_sampling_params(
        self, max_types, shuffle_types, random_drop, max_neg_type_ratio, max_len
    ):
        """
        Sets sampling parameters on the given model.

        Parameters:
        - model: The model object to update.
        - max_types: Maximum types parameter.
        - shuffle_types: Boolean indicating whether to shuffle types.
        - random_drop: Boolean indicating whether to randomly drop elements.
        - max_neg_type_ratio: Maximum negative type ratio.
        - max_len: Maximum length parameter.
        """
        self.config.max_types = max_types
        self.config.shuffle_types = shuffle_types
        self.config.random_drop = random_drop
        self.config.max_neg_type_ratio = max_neg_type_ratio
        self.config.max_len = max_len

    def prepare_state_dict(self, state_dict):
        """
        Prepare state dict in the case of torch.compile
        """
        new_state_dict = {}
        for key, tensor in state_dict.items():
            key = re.sub(r"_orig_mod\.", "", key)
            new_state_dict[key] = tensor
        return new_state_dict

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[GLiNERConfig] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        safe_serialization = False,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            safe_serialization (`bool`):
                Whether to save the model using `safetensors` or the traditional way for PyTorch.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        model_state_dict = self.prepare_state_dict(self.model.state_dict())
        # save model weights using safetensors
        if safe_serialization:
            save_file(model_state_dict, os.path.join(save_directory, "model.safetensors"))
        else:
            torch.save(
                model_state_dict,
                os.path.join(save_directory, "pytorch_model.bin"),
            )

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            config.to_json_file(save_directory / "gliner_config.json")

        self.data_processor.transformer_tokenizer.save_pretrained(save_directory)
        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        load_tokenizer: Optional[bool] = False,
        resize_token_embeddings: Optional[bool] = True,
        load_onnx_model: Optional[bool] = False,
        onnx_model_file: Optional[str] = "model.onnx",
        compile_torch_model: Optional[bool] = False,
        session_options: Optional[ort.SessionOptions] = None,
        _attn_implementation: Optional[str] = None,
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        **model_kwargs,
    ):
        """
        Load a pretrained model from a given model ID.

        Args:
            model_id (str): Identifier of the model to load.
            revision (Optional[str]): Specific model revision to use.
            cache_dir (Optional[Union[str, Path]]): Directory to store downloaded models.
            force_download (bool): Force re-download even if the model exists.
            proxies (Optional[Dict]): Proxy configuration for downloads.
            resume_download (bool): Resume interrupted downloads.
            local_files_only (bool): Use only local files, don't download.
            token (Union[str, bool, None]): Token for API authentication.
            map_location (str): Device to map model to. Defaults to "cpu".
            strict (bool): Enforce strict state_dict loading.
            load_tokenizer (Optional[bool]): Whether to load the tokenizer. Defaults to False.
            resize_token_embeddings (Optional[bool]): Resize token embeddings. Defaults to True.
            load_onnx_model (Optional[bool]): Load ONNX version of the model. Defaults to False.
            onnx_model_file (Optional[str]): Filename for ONNX model. Defaults to 'model.onnx'.
            compile_torch_model (Optional[bool]): Compile the PyTorch model. Defaults to False.
            session_options (Optional[onnxruntime.SessionOptions]): ONNX Runtime session options. Defaults to None.
            **model_kwargs: Additional keyword arguments for model initialization.

        Returns:
            An instance of the model loaded from the pretrained weights.
        """
        # Newer format: Use "pytorch_model.bin" and "gliner_config.json"
        model_dir = Path(model_id)  # / "pytorch_model.bin"
        if not model_dir.exists():
            model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        model_file = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, "pytorch_model.bin")
        config_file = Path(model_dir) / "gliner_config.json"

        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
        else:
            if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
                tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
            else:
                tokenizer = None
        with open(config_file, "r") as f:
            config_ = json.load(f)
        config = GLiNERConfig(**config_)

        if _attn_implementation is not None:
            config._attn_implementation = _attn_implementation
            
        if max_length is not None:
            config.max_len = max_length
        if max_width is not None:
            config.max_width = max_width
        if post_fusion_schema is not None:
            config.post_fusion_schema = post_fusion_schema
            print('Post fusion is set.')

        add_tokens = ["[FLERT]", config.ent_token, config.sep_token]

        if not load_onnx_model:
            gliner = cls(config, tokenizer=tokenizer, encoder_from_pretrained=False, cache_dir=cache_dir)
            # to be able to load GLiNER models from previous version
            if (
                config.class_token_index == -1 or config.vocab_size == -1
            ) and resize_token_embeddings and not config.labels_encoder:
                gliner.resize_token_embeddings(add_tokens=add_tokens)
            if model_file.endswith("safetensors"):
                state_dict = {}
                with safe_open(model_file, framework="pt", device=map_location) as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location=torch.device(map_location), weights_only=True)
            gliner.model.load_state_dict(state_dict, strict=strict)
            gliner.model.to(map_location)
            if compile_torch_model and "cuda" in map_location:
                print("Compiling torch model...")
                gliner.compile()
            elif compile_torch_model:
                warnings.warn(
                    "It's not possible to compile this model putting it to CPU, you should set `map_location` to `cuda`."
                )
            gliner.eval()
        else:
            model_file = Path(model_dir) / onnx_model_file
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"The ONNX model can't be loaded from {model_file}."
                )
            if session_options is None:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
            providers = ['CPUExecutionProvider']
            if "cuda" in map_location:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available but `map_location` is set to 'cuda'.")
                providers = ['CUDAExecutionProvider']
            ort_session = ort.InferenceSession(model_file, session_options, providers=providers)
            if config.span_mode == "token_level":
                model = TokenORTModel(ort_session)
            else:
                model = SpanORTModel(ort_session)

            gliner = cls(config, tokenizer=tokenizer, model=model)
            if (
                config.class_token_index == -1 or config.vocab_size == -1
            ) and resize_token_embeddings:
                gliner.data_processor.transformer_tokenizer.add_tokens(add_tokens)

        if (len(gliner.data_processor.transformer_tokenizer)!=gliner.config.vocab_size
                                                        and gliner.config.vocab_size!=-1):
            new_num_tokens = len(gliner.data_processor.transformer_tokenizer)
            model_embeds = gliner.model.token_rep_layer.resize_token_embeddings(
                new_num_tokens, None
            )
        return gliner

    @staticmethod
    def load_from_config(gliner_config: GLiNERConfig):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            gliner_config.model_name,
            model_max_length=gliner_config.max_len
        )

        # Add special tokens and update config
        gliner_config.class_token_index = len(tokenizer)
        tokenizer.add_tokens([
            gliner_config.ent_token,
            gliner_config.sep_token
        ])
        gliner_config.vocab_size = len(tokenizer)

        # Select appropriate processor
        words_splitter = WordsSplitter()
        if gliner_config.span_mode == "token_level":
            data_processor = TokenProcessor(
                gliner_config,
                tokenizer,
                words_splitter,
                preprocess_text=True
            )
        else:
            data_processor = SpanProcessor(
                gliner_config,
                tokenizer,
                words_splitter,
                preprocess_text=True
            )

        # Instantiate model and apply token resizing
        model = GLiNER(
            gliner_config,
            data_processor=data_processor
        )

        model.resize_token_embeddings(
            [gliner_config.ent_token, gliner_config.sep_token],
            set_class_token_index=False,
            add_tokens_to_tokenizer=False
        )

        return model