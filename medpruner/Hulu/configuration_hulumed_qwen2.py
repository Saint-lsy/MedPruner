"""HuluMed model configuration."""

import importlib.util
import os.path as osp
from typing import Optional, Dict, Any

from transformers import AutoConfig, AutoModel, PretrainedConfig, Qwen2Config

try:
    from visionzip.models.Hulu.configuration_hulumed_encoder import HulumedVisionEncoderConfig
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "configuration_hulumed_encoder",
        osp.join(osp.dirname(__file__), "configuration_hulumed_encoder.py"),
    )
    configuration_hulumed_encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration_hulumed_encoder)
    HulumedVisionEncoderConfig = getattr(
        configuration_hulumed_encoder,
        "HulumedVisionEncoderConfig",
    )

try:
    from visionzip.models.Hulu.modeling_hulumed_encoder import HulumedVisionEncoderModel
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "modeling_hulumed_encoder",
        osp.join(osp.dirname(__file__), "modeling_hulumed_encoder.py"),
    )
    modeling_hulumed_encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling_hulumed_encoder)
    HulumedVisionEncoderModel = getattr(
        modeling_hulumed_encoder,
        "HulumedVisionEncoderModel",
    )

AutoConfig.register("hulumed_vision_encoder", HulumedVisionEncoderConfig)
AutoModel.register(HulumedVisionEncoderConfig, HulumedVisionEncoderModel)


class HulumedQwen2Config(Qwen2Config):
    """
    HuluMed model configuration.
    
    This configuration class extends Qwen2Config to store the configuration of a HuluMed model.
    It includes configuration for the vision encoder and multimodal projector.
    """

    model_type = "hulumed_qwen2"
    sub_configs = {"vision_encoder_config": HulumedVisionEncoderConfig}

    def __init__(
        self,
        vision_encoder: Optional[str] = None,
        vision_encoder_config: Dict[str, Any] = {}, 
        mm_projector_type: str = "mlp2x_gelu",
        use_token_compression: bool = True,
        image_token_index: int = -1,
        **kwargs,
    ):
        """
        Initialize HuluMed configuration.
        
        Args:
            vision_encoder (str, optional): Path or identifier of the vision encoder.
            vision_encoder_config (dict, optional): Configuration for the vision encoder.
            mm_projector_type (str): Type of multimodal projector. Default is "mlp2x_gelu".
            use_token_compression (bool): Whether to use token compression for videos. Default is True.
            image_token_index (int): Token index for image placeholders. Default is -1.
            **kwargs: Additional arguments passed to Qwen2Config.
        """
        super().__init__(**kwargs)
        self.model_type = "hulumed_qwen2"

        self.vision_encoder = vision_encoder
        
        if vision_encoder_config is not None and not isinstance(vision_encoder_config, PretrainedConfig):
            vision_encoder_config = HulumedVisionEncoderConfig(**vision_encoder_config)
        
        self.vision_encoder_config = vision_encoder_config
        self.mm_projector_type = mm_projector_type
        self.use_token_compression = use_token_compression
        self.image_token_index = image_token_index