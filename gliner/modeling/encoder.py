import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from .layers import LayersFuser
from ..utils import is_module_available, MissedPackageException
from ..infer_packing import InferencePackingConfig, pack_requests, unpack_spans
from typing import Optional, Union, List

IS_LLM2VEC = is_module_available('llm2vec')
IS_PEFT = is_module_available('peft')
IS_TURBOT5 = is_module_available('turbot5')
IS_FLASHDEBERTA = is_module_available('flashdeberta')

if IS_LLM2VEC:
    from llm2vec.models import MistralBiModel, LlamaBiModel, GemmaBiModel, Qwen2BiModel
    DECODER_MODEL_MAPPING = {
        "MistralConfig": MistralBiModel,
        "LlamaConfig": LlamaBiModel,
        "GemmaConfig": GemmaBiModel,
        "Qwen2Config": Qwen2BiModel
    }
else:
    DECODER_MODEL_MAPPING = {}

if IS_TURBOT5:
    from turbot5.model.modeling import T5EncoderModel
else:
    from transformers import T5EncoderModel

if IS_FLASHDEBERTA:
    from flashdeberta import FlashDebertaV2Model as DebertaV2Model
else:
    from transformers import DebertaV2Model

if IS_PEFT:
    from peft import LoraConfig, get_peft_model

class Transformer(nn.Module):
    def __init__(
        self, 
        model_name, 
        config, 
        from_pretrained=False, 
        labels_encoder = False, 
        cache_dir:Optional[Union[str, Path]] = None
    ):
        super().__init__()
        if labels_encoder:
            encoder_config = config.labels_encoder_config
        else:
            encoder_config = config.encoder_config
        if encoder_config is None:
            encoder_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            if config.vocab_size!=-1:
                encoder_config.vocab_size = config.vocab_size

        if config._attn_implementation is not None and not labels_encoder:
            encoder_config._attn_implementation = config._attn_implementation

        config_name = encoder_config.__class__.__name__

        kwargs = {}
        if config_name in DECODER_MODEL_MAPPING:
            if not IS_LLM2VEC:
                raise MissedPackageException(f"The llm2vec package must be installed to use this decoder model: {config_name}")
            else:
                print('Loading decoder model using LLM2Vec...')
                ModelClass = DECODER_MODEL_MAPPING[config_name]
            custom = True
        elif config_name in {'T5Config', 'MT5Config'}:
            custom = True
            ModelClass = T5EncoderModel
            if IS_TURBOT5:
                kwargs = {"attention_type": 'flash'}
        elif config_name in {'DebertaV2Config'}:
            custom = True
            ModelClass = DebertaV2Model
        else:
            custom = False
            ModelClass = AutoModel

        if from_pretrained:
            self.model = ModelClass.from_pretrained(model_name, trust_remote_code=True)
        else:
            if not custom:
                self.model = ModelClass.from_config(encoder_config, trust_remote_code=True)
            else:
                self.model = ModelClass(encoder_config, **kwargs)

        adapter_config_file = Path(model_name) / "adapter_config.json"

        if adapter_config_file.exists():
            if not IS_PEFT:
                warnings.warn(f"Adapter configs were detected, if you want to apply them you need to install peft package.")
            else:
                adapter_config = LoraConfig.from_pretrained(model_name)
                self.model = get_peft_model(self.model, adapter_config)

        if config.fuse_layers:
            self.layers_fuser = LayersFuser(encoder_config.num_hidden_layers,
                                                        encoder_config.hidden_size)

        if labels_encoder:
            config.labels_encoder_config = encoder_config
        else:
            config.encoder_config = encoder_config

        self.config = config

    def forward(self, *args, **kwargs):
        if self.config.fuse_layers:
            output_hidden_states = True
        else:
            output_hidden_states = False
        output = self.model(*args, output_hidden_states = output_hidden_states, 
                                            return_dict = True,  **kwargs)
        if self.config.fuse_layers:
            encoder_layer = self.layers_fuser(output.hidden_states)
        else:
            encoder_layer = output[0]

        return encoder_layer
    
class Encoder(nn.Module):
    def __init__(self, config, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]]= None):
        super().__init__()

        self.bert_layer = Transformer( #transformer_model
            config.model_name, config, from_pretrained, cache_dir = cache_dir
        )

        bert_hidden_size = self.bert_layer.model.config.hidden_size

        if config.hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, config.hidden_size)

    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        return self.bert_layer.model.resize_token_embeddings(new_num_tokens, 
                                                                pad_to_multiple_of)

    def get_input_embeddings(self):
        return self.bert_layer.model.get_input_embeddings()
    
    def encode_text(self, input_ids, attention_mask, *args, **kwargs):
        packing_config: Optional[InferencePackingConfig] = kwargs.pop("packing_config", None)
        pair_attention_mask = kwargs.pop("pair_attention_mask", None)

        if (
            packing_config is not None
            and not self.training
            and pair_attention_mask is None
            and isinstance(input_ids, torch.Tensor)
            and isinstance(attention_mask, torch.Tensor)
            and input_ids.dim() == 2
        ):
            token_embeddings = self._encode_with_packing(
                input_ids,
                attention_mask,
                packing_config,
                *args,
                **kwargs,
            )
        else:
            attn_to_use = pair_attention_mask if pair_attention_mask is not None else attention_mask
            if attn_to_use is not None and isinstance(attn_to_use, torch.Tensor) and attn_to_use.dim() == 3:
                attn_to_use = attn_to_use.to(torch.bool)
            token_embeddings = self.bert_layer(
                input_ids,
                attn_to_use,
                *args,
                **kwargs,
            )

        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings

    def _encode_with_packing(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packing_config: InferencePackingConfig,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        lengths = attention_mask.sum(dim=-1).tolist()
        seq_len = int(input_ids.size(1))
        if not lengths or all(int(l) == seq_len for l in lengths):
            return self.bert_layer(input_ids, attention_mask, *args, **kwargs)

        requests = []
        for row, length in zip(input_ids, lengths):
            length = int(length)
            if length <= 0:
                requests.append({"input_ids": []})
            else:
                requests.append({"input_ids": row[:length].tolist()})

        pad_token_id = self.bert_layer.model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        packed = pack_requests(requests, packing_config, pad_token_id)

        device = input_ids.device
        packed_ids = packed.input_ids.to(device=device)
        packed_mask = packed.pair_attention_mask.to(device=device)
        packed_fallback = packed.attention_mask.to(device=device)

        attn_to_use = packed_mask if packed_mask.numel() else packed_fallback
        token_embeddings = self.bert_layer(
            packed_ids,
            attn_to_use,
            *args,
            **kwargs,
        )

        unpacked: List[torch.Tensor] = unpack_spans(token_embeddings, packed)
        hidden_size = token_embeddings.size(-1)
        batch, seq = input_ids.size()
        output = token_embeddings.new_zeros(batch, seq, hidden_size)
        for idx, target in enumerate(unpacked):
            tgt_len = int(target.size(0))
            if tgt_len == 0:
                continue
            output[idx, :tgt_len] = target
        return output
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        token_embeddings = self.encode_text(*args, **kwargs)
        return token_embeddings

class BiEncoder(Encoder):
    def __init__(self, config, from_pretrained: bool = False, cache_dir:Optional[Union[str, Path]] = None):
        super().__init__(config, from_pretrained)
        if config.labels_encoder is not None:
            self.labels_encoder = Transformer( #transformer_model
                config.labels_encoder, config, from_pretrained, True, cache_dir=cache_dir
            )
            le_hidden_size = self.labels_encoder.model.config.hidden_size

            if config.hidden_size != le_hidden_size:
                self.labels_projection = nn.Linear(le_hidden_size, config.hidden_size)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_labels(self, input_ids, attention_mask, *args, **kwargs):
        labels_embeddings = self.labels_encoder(input_ids, attention_mask, *args, **kwargs)
        if hasattr(self, "labels_projection"):
            labels_embeddings = self.labels_projection(labels_embeddings)
        labels_embeddings = self.mean_pooling(labels_embeddings, attention_mask)
        return labels_embeddings

    def forward(self, input_ids, attention_mask, 
                    labels_input_ids = None, labels_attention_mask=None, 
                                            *args, **kwargs) -> torch.Tensor:
        token_embeddings = self.encode_text(input_ids, attention_mask, *args, **kwargs)

        labels_embeddings = self.encode_labels(labels_input_ids, labels_attention_mask, *args, **kwargs)
        return token_embeddings, labels_embeddings