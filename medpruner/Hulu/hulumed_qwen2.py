# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""PyTorch HuluMed model."""

import importlib.util
import os.path as osp
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoModel
from .modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from ..utils import get_mask_entropy

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# Image arguments
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Video arguments
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 128
MAX_FRAMES = 768
NUM_FRAMES_PER_SECOND = 1

# Audio arguments
AUDIO_TOKEN_INDEX = -202
DEFAULT_AUDIO_TOKEN = "<audio>"

# Stream arguments
STREAM_START_TOKEN = "<|stream_start|>"
STREAM_END_TOKEN = "<|stream_end|>"
STREAM_MAX_FRAMES = 400

MODAL_INDEX_MAP = {
    "<image>": -200,
    "<video>": -201,
    "<audio>": -202,
}

subimage_token_num=196
try:
    from medpruner.Hulu.configuration_hulumed_qwen2 import HulumedQwen2Config
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "configuration_hulumed_qwen2",
        osp.join(osp.dirname(__file__), "configuration_hulumed_qwen2.py"),
    )
    configuration_hulumed_qwen2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration_hulumed_qwen2)
    HulumedQwen2Config = getattr(
        configuration_hulumed_qwen2,
        "HulumedQwen2Config",
    )


def build_mlp(depth, hidden_size, output_hidden_size):
    """Build MLP layers for projection."""
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


def build_vision_projector(config, delay_load=False, **kwargs):
    """Build vision projector based on config."""
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    
    if projector_type == "linear":
        return nn.Linear(config.vision_encoder_config.hidden_size, config.hidden_size)
    elif projector_type.startswith("mlp"):
        return MlpGeluProjector(config, projector_type)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')


class MlpGeluProjector(nn.Module):
    """MLP projector with GELU activation."""

    def __init__(self, config, projector_type):
        super().__init__()
        
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match is None:
            raise ValueError(f"Invalid projector type format: {projector_type}")
        mlp_depth = int(mlp_gelu_match.group(1))

        self.readout = build_mlp(
            mlp_depth, 
            config.vision_encoder_config.hidden_size, 
            config.hidden_size
        )

    def forward(self, x):
        return self.readout(x)


class HulumedMetaModel:
    """Meta model for HuluMed that handles vision encoder initialization."""

    def __init__(self, config):
        super(HulumedMetaModel, self).__init__(config)
        print('config.vision_encoder',config.vision_encoder)
        if config.vision_encoder is not None:
            # Load from pretrained path
            print('Load from pretrained path')
            self.vision_encoder = AutoModel.from_pretrained(
                config.vision_encoder,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
            self.config.vision_encoder_config = self.vision_encoder.config
            self.config.vision_encoder = None
        elif config.vision_encoder_config is not None:
            # Build from config
            print('Build from config')
            self.vision_encoder = AutoModel.from_config(
                self.config.vision_encoder_config,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
        else:
            raise ValueError("Vision encoder is not provided in config")
        
        self.mm_projector = build_vision_projector(config)

    def get_vision_encoder(self):
        return self.vision_encoder

    def get_mm_projector(self):
        return self.mm_projector


class HulumedQwen2Model(HulumedMetaModel, Qwen2Model):
    """HuluMed Qwen2 Model."""
    
    config_class = HulumedQwen2Config

    def __init__(self, config: HulumedQwen2Config):
        super(HulumedQwen2Model, self).__init__(config)


class HulumedMetaForCausalLM(ABC):
    """Meta class for HuluMed Causal LM with multimodal support."""

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Encode images using vision encoder and projector."""
        mm_features,logits, attn_key = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        mm_features = self.get_model().mm_projector(mm_features)

        return mm_features,logits, attn_key

    def _get_valid_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
    ):
        """Filter out text-only samples and keep only valid visual tokens."""
        valid_masks = []
        for num_patches, modal in zip(batched_num_patches, modals):
            valid_mask = torch.full(
                (num_patches,), 
                modal != "text", 
                dtype=torch.bool, 
                device=mm_features.device
            )
            valid_masks.append(valid_mask)
        mm_features = mm_features[torch.cat(valid_masks)]
        return mm_features

    def _maybe_truncate_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        compression_mask: torch.BoolTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """Truncate visual tokens if necessary based on position_ids."""
        if position_ids is None or mm_features.shape[0] == input_ids.eq(self.config.image_token_index).sum():
            return mm_features, compression_mask

        truncation_mask = []
        for num_patches, modal in zip(batched_num_patches, modals):
            if modal == "text":
                truncation_mask.append(torch.ones((0,), dtype=torch.bool, device=input_ids.device))
            else:
                truncation_mask.append(torch.ones((num_patches,), dtype=torch.bool, device=input_ids.device))

        seq_end_indices = torch.nonzero(position_ids == 0)[:, 0]
        seq_end_indices = seq_end_indices[seq_end_indices > 0].tolist() + [len(input_ids)]
        seq_start_indices = [0] + seq_end_indices[:-1]
        num_visual_tokens = [
            input_ids[start:end].eq(self.config.image_token_index).sum()
            for start, end in zip(seq_start_indices, seq_end_indices)
        ]

        for n, mask in zip(num_visual_tokens, truncation_mask):
            if len(mask) > 0:
                mask[n:] = False
        truncation_mask = torch.cat(truncation_mask)

        return mm_features[truncation_mask], compression_mask[truncation_mask]

    def _get_compression_mask(
        self,
        pixel_values: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        modals: List[str],
        threshold: float = 0.1,
        min_tokens: int = 1,
    ) -> torch.BoolTensor:
        """Get compression mask for video tokens based on frame differences."""
        batched_images = pixel_values.split(grid_sizes.prod(dim=1).tolist(), dim=0)
        compression_masks = []

        for images, num_patches, grid_size, merge_size, modal in zip(
            batched_images, batched_num_patches, grid_sizes, merge_sizes, modals
        ):
            t, h, w = grid_size
            if modal == "image" or (modal == "video" and t == 1):
                compression_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=images.device))

            elif modal == "video":
                # Video token compression based on pixel differences
                images = images.view(t, (h // merge_size) * (w // merge_size), -1)  # (t, P, D)
                if self.slice_compose:
                    # Cumulative reference frame + whole-frame L1 distance
                    mask = []
                    ref_index = 0
                    for i in range(1,t):
                        frame_l1 = torch.abs(images[i] - images[ref_index]).mean()
                        # Keep frame if whole-frame L1 distance > gamma
                        if frame_l1 > self.gamma:
                            ref_index = i
                            img_mask = torch.ones(images.shape[1], dtype=torch.bool, device=images.device)
                        else:
                            img_mask = torch.zeros(images.shape[1], dtype=torch.bool, device=images.device)
                        mask.append(img_mask)

                    stacked_masks = torch.stack(mask, dim=0)
                    
                    # Prepend first image (always kept by default)
                    first_mask = torch.ones(images.shape[1], dtype=torch.bool, device=images.device).unsqueeze(0)
                    mask = torch.cat([first_mask, stacked_masks], dim=0)
                else:
                    pixel_diff = images[1:] - images[:-1]
                    pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                    pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                    mask = (pixel_diff / 255.0) > threshold

                # 7) ensure at least min_tokens per frame
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                mask[padding_ids, :min_tokens] = 1
                compression_masks.append(mask.flatten())

            else:
                # Pseudo image case
                compression_masks.append(torch.ones((0,), dtype=torch.bool, device=images.device))

        return torch.cat(compression_masks)

    def _compress_visual_tokens(
        self,
        compression_mask: torch.BoolTensor,
        mm_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """Compress visual tokens based on compression mask."""
        mm_features = mm_features[compression_mask]
        image_selected = (input_ids == self.config.image_token_index)

        text_masks = torch.logical_not(image_selected)
        text_masks[image_selected] = compression_mask
        input_ids = input_ids[text_masks]

        if attention_mask is not None:
            attention_mask = attention_mask[text_masks]
        if labels is not None:
            labels = labels[text_masks]
        if position_ids is not None:
            position_ids = position_ids[text_masks]
            pos_start = [0] + torch.nonzero(position_ids == 0)[:, 0].tolist()
            pos_end = pos_start[1:] + [len(input_ids)]
            position_ids = torch.cat([
                torch.arange(end - start, device=input_ids.device) 
                for start, end in zip(pos_start, pos_end)
            ])

        return mm_features, input_ids, attention_mask, position_ids, labels

    def dyn_token_compose(
        self,
        mm_features: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        attn_logits: torch.FloatTensor,
        attn_keys: torch.FloatTensor,
        merge_sizes :torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """
        Apply dyn_token token selection based on attention scores.

        Selects dominant tokens (65%) by attention weight and contextual tokens (5%)
        by similarity clustering, then aggregates remaining tokens.

        Args:
            mm_features: Visual features [num_tokens, hidden_dim]
            attn_logits: Attention logits [num_tokens]
            attn_key: Key states [num_heads, num_tokens, head_dim]

        Returns:
            selected_features: Selected visual features with aggregated contextual tokens
            selection_mask: Boolean mask for selected tokens (True = keep)
        """
        batched_images = mm_features.split(batched_num_patches.tolist(), dim=0)

        compression_masks = []
        select_masks = []

        for mm_feature,attn_logit,attn_key,merge_size in zip(batched_images,attn_logits,attn_keys,merge_sizes):

            with torch.no_grad():
                attn_mean = attn_logit.mean(dim=0)  

                attn_mean = attn_mean.sum(dim=0) 
                attn_mean = attn_mean.view(attn_mean.shape[0] // merge_size**2, -1).mean(dim=-1)

                attn_key = attn_key.view(attn_key.shape[0], attn_key.shape[1] // merge_size**2, merge_size**2, attn_key.shape[-1]).mean(dim=2)
                attn_key = attn_key.mean(dim=0).unsqueeze(0)
        
            num_tokens = attn_mean.size(0)
            device = attn_mean.device

            # Calculate selection ratios
            attn_probs = torch.nn.functional.softmax(attn_mean, dim=-1)
            contextual_num = max(int(0.05 * num_tokens), 1)
            topk_indices = get_mask_entropy(attn_probs, tau=self.tau)

            select_mask = torch.zeros_like(attn_mean, dtype=torch.bool)
            select_mask[topk_indices] = True

            # Step 2: Process contextual tokens (non-dominant tokens)
            contextual_mask = ~select_mask
            metric_filtered = attn_key[:, contextual_mask]
            # Output attention from the last layer
            # self.last_attn_mean = metric_filtered.clone().detach()
            metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)
            del attn_key, metric_filtered  # Free memory

            # Step 3: Select contextual token targets via uniform sampling
            step = max(1, metric_normalized.shape[1] // contextual_num)
            target_indices = torch.arange(0, metric_normalized.shape[1], step, device=device)[:contextual_num]
            target_tokens = metric_normalized[:, target_indices, :]

            # Step 4: Aggregate non-contextual tokens into targets
            tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=device), target_indices), :]
            
            similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
            assign_one_hot = torch.zeros(
                tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num,
                dtype=attn_mean.dtype, device=device
            )
            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)

            # Step 5: Combine dominant and contextual token masks
            false_pos = (~select_mask).nonzero(as_tuple=True)[0]
            select_mask[false_pos[target_indices]] = True

            # Step 6: Aggregate token features
            # Get indices in the original mm_features space
            contexual_input_idx = false_pos[target_indices]

            # Aggregate non-selected tokens into contextual tokens
            hidden_states_filtered = mm_feature[contextual_mask]  # All non-dominant tokens
            hidden_to_merge = hidden_states_filtered[~torch.isin(
                torch.arange(hidden_states_filtered.shape[0], device=device),
                target_indices
            )]
            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge.unsqueeze(0)).squeeze(0) / counts.squeeze(0)
            target_hidden = hidden_states_filtered[target_indices]  # Contextual target tokens

            contextual_token = target_hidden + aggregated_hidden

            # Step 7: Build output features
            # Only update contextual anchor tokens
            selected_feature = mm_feature.clone()
            selected_feature[contexual_input_idx] = contextual_token
            
            compression_masks.append(selected_feature)
            select_masks.append(select_mask)
            
            del contextual_token, hidden_states_filtered, hidden_to_merge, aggregated_hidden
            
        return torch.cat(compression_masks),torch.cat(select_masks)
    
    @staticmethod
    def cal_compose_rate(mask):
        return mask.sum()/(mask.shape[0])

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
    ):
        """Prepare inputs and labels for multimodal training/inference."""
        vision_encoder = self.get_vision_encoder()

        # Text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels

        # 1. Flatten text inputs
        B, N = input_ids.shape
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. Embed visual tokens
        batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()
        mm_features, attn_logits, attn_key = self.encode_images(pixel_values, grid_sizes, merge_sizes)
        mm_features = mm_features.to(input_ids.device)
        mm_features = self._get_valid_visual_tokens(mm_features, batched_num_patches, modals)

        
        # Token compression based on inter-frame differences
        compression_mask = torch.ones(batched_num_patches.sum(), dtype=torch.bool,device=input_ids.device)
        dyn_token_mask = torch.ones(batched_num_patches.sum(), dtype=torch.bool,device=input_ids.device)
        
        compression_mask = self._get_compression_mask(
            pixel_values, batched_num_patches, grid_sizes, merge_sizes, modals
        )
        mm_features, compression_mask = self._maybe_truncate_visual_tokens(
            mm_features, compression_mask, batched_num_patches, modals, input_ids, position_ids
        )
        if self.token_compose:
            grid_sizes_ = []
            merge_sizes_ = []
            for grid_size,merge_size in zip(grid_sizes,merge_sizes):
                dim = grid_size[0].item()
                merge_sizes_.append(merge_size.unsqueeze(0).expand(dim,1))
                tmp = grid_size.unsqueeze(0).expand(dim,-1)
                tmp[:,0] = 1
                grid_sizes_.append(tmp)
            grid_sizes_ = torch.cat(grid_sizes_)
            merge_sizes_ = torch.cat(merge_sizes_)

            batched_num_patches_ = grid_sizes_.prod(dim=1).div(merge_sizes ** 2).long()
            mm_features, dyn_token_mask = self.dyn_token_compose(
                mm_features,batched_num_patches_, attn_logits, attn_key,merge_sizes_
            )

        combined_mask = compression_mask & dyn_token_mask
        self.compression_rate = self.cal_compose_rate(compression_mask)
        self.dyn_token_rate = self.cal_compose_rate(dyn_token_mask)
        self.select_rate = self.cal_compose_rate(combined_mask)

        # 3. Apply dyn_token token selection if enabled (combines with temporal compression)
        assert B == 1, "Token compression is only supported for batch_size=1"
        # 3c. Compress tokens using combined mask
        mm_features, input_ids, attention_mask, position_ids, labels = self._compress_visual_tokens(
            combined_mask, mm_features, input_ids, attention_mask, position_ids, labels
        )

        # 4. Embed text tokens
        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()

        # 5. Replace multimodal tokens with features
        image_selected = (input_ids == self.config.image_token_index)
        inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features

        # 6. Reshape back to batched format
        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels


class HulumedQwen2ForCausalLM(Qwen2ForCausalLM, HulumedMetaForCausalLM):
    """HuluMed Qwen2 model for causal language modeling with multimodal support."""
    
    config_class = HulumedQwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = HulumedQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.slice_compose = kwargs.pop("slice_compose", True)
        self.token_compose = kwargs.pop("token_compose", True)
        self.tau = kwargs.pop("tau", 0.9)
        self.gamma = kwargs.pop("gamma", 0.05)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        # Multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass with multimodal support."""
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )
        torch.cuda.empty_cache()
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **loss_kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        # Multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate with multimodal support."""
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for generation."""
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs