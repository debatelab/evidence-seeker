"PreprocessingConfig"

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
import importlib.resources as pkg_resources

import pydantic
import yaml
from loguru import logger
from llama_index.core import ChatPromptTemplate

from evidence_seeker.backend import GuidanceType


@lru_cache(maxsize=1)
def _load_default_config_dict() -> Dict[str, Any]:
    """Load default configuration from YAML. Cached for performance."""
    try:
        # Use importlib.resources to access package data
        config_file = pkg_resources.files(
            "evidence_seeker.package_data"
        ).joinpath("config/preprocessing_config.yaml")

        with config_file.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, AttributeError):
        logger.error(
            "Failed to load default preprocessing configuration from package data. "
            "Ensure 'evidence_seeker.package_data' is properly installed."
        )
        raise


def _get_default_for_field(field_name: str) -> Any:
    """Get default value for a specific field from YAML."""
    config_dict = _load_default_config_dict()
    return config_dict.get(field_name)


class PreprocessorModelStepConfig(pydantic.BaseModel):
    prompt_template: str
    system_prompt: str | None = None
    # Fields used for constrained decoding
    guidance_type: Optional[str] = None

    @pydantic.field_validator('guidance_type')
    @classmethod
    def validate_guidance_type(cls, v):
        allowed_values = {GuidanceType.PYDANTIC.value}
        if (v is not None) and (v not in allowed_values):
            raise ValueError(
                f'guidance_type must be one of {allowed_values}, got {v}'
            )
        return v


class PreprocessorStepConfig(pydantic.BaseModel):
    name: str
    description: str | None = None
    used_model_key: str | None = None
    llm_specific_configs: Dict[str, PreprocessorModelStepConfig] = dict()


class ClaimPreprocessingConfig(pydantic.BaseModel):
    config_version: str = pydantic.Field(
        default_factory=lambda: _get_default_for_field("config_version") or "v0.1"
    )
    description: str = pydantic.Field(
        default_factory=lambda: (
            _get_default_for_field("description") or
            "Configuration of EvidenceSeeker's preprocessing component."
        )
    )
    system_prompt: str = pydantic.Field(
        default_factory=lambda: _get_default_for_field("system_prompt")
    )
    language: str = pydantic.Field(
        default_factory=lambda: _get_default_for_field("language") or "DE"
    )
    timeout: int = pydantic.Field(
        default_factory=lambda: _get_default_for_field("timeout") or 900
    )
    # Whether or not the workflow/pipeline should print additional informative messages
    # during execution.
    verbose: bool = pydantic.Field(
        default_factory=lambda: _get_default_for_field("verbose") or False
    )
    env_file: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("env_file") or None
    )

    @pydantic.model_validator(mode='after')
    def load_env_file(self) -> 'ClaimPreprocessingConfig':
        if self.env_file is None:
            logger.warning(
                "No environment file with API keys specified for preprocessor."
                " Please set 'env_file' to a valid path if you want "
                "to load environment variables from a file."
            )
        else:
            # check if the env file exists
            from os import path
            if not path.exists(self.env_file):
                err_msg = (
                    f"Environment file '{self.env_file}' does not exist. "
                    "Please provide a valid path to the environment file. "
                    "Or set it to None if you don't need it and set the "
                    "API keys in other ways as environment variables."
                )
                logger.warning(err_msg)
            else:
                # load the env file
                from dotenv import load_dotenv
                load_dotenv(self.env_file)
            logger.info(
                f"Loaded environment variables from '{self.env_file}'"
            )

        return self

    used_model_key: str
    freetext_descriptive_analysis: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig.model_validate(
            _get_default_for_field("freetext_descriptive_analysis")
        )
    )
    list_descriptive_statements: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig.model_validate(
            _get_default_for_field("list_descriptive_statements")
        )
    )
    freetext_ascriptive_analysis: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig.model_validate(
            _get_default_for_field("freetext_ascriptive_analysis")
        )
    )
    list_ascriptive_statements: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig.model_validate(
            _get_default_for_field("list_ascriptive_statements")
        )
    )
    freetext_normative_analysis: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig.model_validate(
            _get_default_for_field("freetext_normative_analysis")
        )
    )
    list_normative_statements: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig.model_validate(
            _get_default_for_field("list_normative_statements")
        )
    )
    negate_claim: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig.model_validate(
            _get_default_for_field("negate_claim")
        )
    )
    models: Dict[str, Dict[str, Any]] = pydantic.Field(
        default_factory=lambda: dict()
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ClaimPreprocessingConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ClaimPreprocessingConfig instance with values from YAML
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)

    # ==helper functions==
    def _step_config(
        self,
        step_config: Optional[PreprocessorStepConfig] = None,
        step_name: Optional[str] = None
    ) -> PreprocessorStepConfig:
        """Internal convenience function."""
        if step_config is None and step_name is None:
            raise ValueError("Either pass a step config or a name of the pipeline step")
        if step_config is None:
            if step_name == "freetext_descriptive_analysis":
                return self.freetext_descriptive_analysis
            elif step_name == "list_descriptive_statements":
                return self.list_descriptive_statements
            elif step_name == "freetext_ascriptive_analysis":
                return self.freetext_ascriptive_analysis
            elif step_name == "list_ascriptive_statements":
                return self.list_ascriptive_statements
            elif step_name == "freetext_normative_analysis":
                return self.freetext_normative_analysis
            elif step_name == "list_normative_statements":
                return self.list_normative_statements
            elif step_name == "negate_claim":
                return self.negate_claim
            else:
                raise ValueError(f"Did not found step config for {step_name}")
        else:
            return step_config

    def get_step_config(
            self,
            step_name: Optional[str] = None,
            step_config: Optional[PreprocessorStepConfig] = None
    ) -> PreprocessorModelStepConfig:
        """Get the model specific step config for the given step name.

        The requested `PreprocessorModelStepConfig` is determined by either
        the provided `step_name` or the provided `step_config`. If both
        are given, the `step_config` is used.
        """
        step_config = self._step_config(step_config, step_name)
        # used model for this step
        if step_config.used_model_key:
            model_key = step_config.used_model_key
        else:
            model_key = self.used_model_key
        # do we have a model-specific config?
        if step_config.llm_specific_configs.get(model_key):
            model_specific_conf = step_config.llm_specific_configs[model_key]
        else:
            if step_config.llm_specific_configs.get("default") is None:
                msg = (
                    f"Default step config for {step_config.name} "
                    "not found in config."
                )
                logger.error(msg)
                raise ValueError(msg)
            model_specific_conf = step_config.llm_specific_configs["default"]
        return model_specific_conf

    def get_chat_template(
            self,
            step_name: Optional[str] = None,
            step_config: Optional[PreprocessorStepConfig] = None
    ) -> ChatPromptTemplate:
        step_config = self._step_config(step_config, step_name)
        model_specific_conf = self.get_step_config(step_config=step_config)
        prompt_template = model_specific_conf.prompt_template

        return ChatPromptTemplate.from_messages(
            [
                ("system", self.get_system_prompt(step_config=step_config)),
                ("user", prompt_template),
            ]
        )

    def get_system_prompt(
            self,
            step_name: Optional[str] = None,
            step_config: Optional[PreprocessorStepConfig] = None
    ) -> str:
        """Get the system prompt for a specific step of the workflow."""
        step_config = self._step_config(step_config, step_name)
        model_specific_conf = self.get_step_config(step_config=step_config)
        if model_specific_conf.system_prompt:
            return model_specific_conf.system_prompt
        else:
            return self.system_prompt

    def get_model_key(
            self,
            step_name: Optional[str] = None,
            step_config: Optional[PreprocessorStepConfig] = None
    ) -> str:
        """Get the model key for a specific step of the workflow."""
        step_config = self._step_config(step_config, step_name)
        if step_config.used_model_key:
            return step_config.used_model_key
        else:
            return self.used_model_key
