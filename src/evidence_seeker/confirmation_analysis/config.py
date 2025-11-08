"confirmation_analysis.py"

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Self
import importlib.resources as pkg_resources

import enum
import pydantic
import re
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
        ).joinpath("config/confirmation_analysis_config.yaml")
        
        with config_file.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, AttributeError):
        logger.error(
            "Failed to load default confirmation analysis configuration "
            "from package data. "
            "Ensure 'evidence_seeker.package_data' is properly installed."
        )
        raise


def _get_default_for_field(field_name: str) -> Any:
    """Get default value for a specific field from YAML."""
    config_dict = _load_default_config_dict()
    return config_dict.get(field_name)


# TODO: validation for `used_model_key`: Check whether corresponding model is defined


class LogProbsType(enum.Enum):
    OPENAI_LIKE = "openai_like"
    ESTIMATE = "estimate"


def json_schema(pattern: str) -> Dict[str, Any]:
    return {
        "properties": {
            "answer": {
                "type": "string",
                "pattern": pattern
            }
        },
        "required": [
            "answer",
        ]
    }

class ConfirmationAnalyzerModelStepConfig(pydantic.BaseModel):
    prompt_template: str
    system_prompt: str | None = None


class MultipleChoiceTaskStepConfig(ConfirmationAnalyzerModelStepConfig):
    """Configuration for the multiple choice task step."""
    prompt_template: str = (
        "Your task is to sum up the results of a rich textual entailment analysis.\n"
        "\n"
        "<TEXT>{evidence_item}</TEXT>\n"
        "\n"
        "<HYPOTHESIS>{statement}</HYPOTHESIS>\n"
        "\n"
        "Our previous analysis has yielded the following result:\n"
        "\n"
        "<RESULT>\n"
        "{freetext_confirmation_analysis}\n"
        "</RESULT>\n"
        "\n"
        "Please sum up this result by deciding which of the following choices is correct. "
        "Just answer with the label of the correct choice.\n"
        "\n"
        "{answer_options}\n"
        "\n"
    )
    # TODO: As set of strings
    answer_options: Optional[List[str]] = [
        "Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.",
        "Contradiction: The TEXT provides evidence that contradicts the HYPOTHESIS.",
        "Neutral: The TEXT neither supports nor contradicts the HYPOTHESIS.",
    ]
    answer_labels: Optional[List[str]] = ["A", "B", "C"]
    claim_option: Optional[str] = (
        "Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS."
    )
    n_repetitions_mcq: int = 1
    # Used as delimiter between an answer label and an answer option
    delim_str: Optional[str] = ""
    # Fields used for constrained decoding
    guidance_type: Optional[str] = GuidanceType.JSON.value
    # log probs
    logprobs_type: Optional[str] = LogProbsType.OPENAI_LIKE.value

    # JSON schema for JSON Guidance
    @pydantic.computed_field
    @property
    def json_schema(self) -> Optional[str | Dict[str, Any]]:
        if (
            self.guidance_type == GuidanceType.JSON.value
            and (self.answer_labels is not None)
            and len(self.answer_labels) != 0
        ):
            return json_schema(pattern=rf"^({'|'.join(self.answer_labels)})$")
        return None
    
    # Regex for Regex Guidance
    @pydantic.computed_field
    @property
    def constrained_decoding_regex(self) -> Optional[str]:
        if (
            self.guidance_type == GuidanceType.REGEX.value
            and (self.answer_labels is not None)
            and len(self.answer_labels) != 0
        ):
            return rf"^({'|'.join(self.answer_labels)})$"
        return None
    
    # Context-free grammar for Grammar Guidance
    @pydantic.computed_field
    @property
    def constrained_decoding_grammar(self) -> Optional[str]:
        if (
            self.guidance_type == GuidanceType.GRAMMAR.value
            and (self.answer_labels is not None)
            and len(self.answer_labels) != 0
        ):
            return rf'root ::= "{'" | "'.join(self.answer_labels)}"'
        return None
    
    # Validation regex for Prompted Guidance
    @pydantic.computed_field
    @property
    def validation_regex(self) -> Optional[str]:
        if (
            self.guidance_type == GuidanceType.PROMPTED.value
            and (self.answer_labels is not None)
            and len(self.answer_labels) != 0
        ):
            return rf"^({'|'.join(self.answer_labels)})$"
        return None

    # Validation of answer labels for JSON, Regex, Grammar and Prompted Guidance
    @pydantic.model_validator(mode='after')
    def check_answer_labels(self) -> Self:
        if self.answer_labels is None or len(self.answer_labels) == 0:
            if self.guidance_type in [GuidanceType.JSON.value, GuidanceType.REGEX.value, GuidanceType.GRAMMAR.value, GuidanceType.PROMPTED.value]:
                raise ValueError(
                    'Please provide possible answer labels for multiple '
                    f'choice tasks when using {self.guidance_type.capitalize()} Guidance.'
                )
        else:
            if self.guidance_type == GuidanceType.JSON.value:
                seen = set()
                seen_twice = set(
                    x for x in self.answer_labels if x in seen or seen.add(x)
                )
                valid_labels = True; [valid_labels := valid_labels & (re.match(r"^[a-zA-Z0-9]$", l) is not None) for l in self.answer_labels]
                if len(seen_twice) != 0 or not valid_labels:
                    raise ValueError(
                        "JSON Guidance assumes unique and single characters "
                        "as answer labels. Possible characters are ASCII "
                        "characters in the ranges of a-z, A-Z and 0-9."
                    )
        return self


class ConfirmationAnalyzerStepConfig(pydantic.BaseModel):
    name: str
    description: str | None = None
    used_model_key: str | None = None
    llm_specific_configs: Dict[
        str, ConfirmationAnalyzerModelStepConfig | MultipleChoiceTaskStepConfig
    ] = dict()


class ConfirmationAnalyzerConfig(pydantic.BaseModel):
    config_version: str = pydantic.Field(
        default_factory=lambda: _get_default_for_field("config_version") or "v0.2"
    )
    description: str = pydantic.Field(
        default_factory=lambda: (
            _get_default_for_field("description") or
            "Configuration of EvidenceSeeker's confirmation analyzer component."
        )
    )
    system_prompt: str = pydantic.Field(
        default_factory=lambda: _get_default_for_field("system_prompt")
    )
    timeout: int = pydantic.Field(
        default_factory=lambda: _get_default_for_field("timeout") or 900
    )
    # Whether or not the workflow/pipeline should print additional informative
    # messages during execution.
    verbose: bool = pydantic.Field(
        default_factory=lambda: _get_default_for_field("verbose") or False
    )
    used_model_key: Optional[str] = pydantic.Field(
        default_factory=lambda: _get_default_for_field("used_model_key") or None
    )
    env_file: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("env_file") or None
    )

    @pydantic.model_validator(mode='after')
    def load_env_file(self) -> 'ConfirmationAnalyzerConfig':
        if self.env_file is None:
            logger.warning(
                "No environment file with API keys specified for confirmation "
                "analyzer. Please set 'env_file' to a valid path if you want "
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

    freetext_confirmation_analysis: ConfirmationAnalyzerStepConfig = pydantic.Field(
        default_factory=lambda: ConfirmationAnalyzerStepConfig.model_validate(
            _get_default_for_field("freetext_confirmation_analysis")
        )
    )
    multiple_choice_confirmation_analysis: ConfirmationAnalyzerStepConfig = pydantic.Field(
        default_factory=lambda: ConfirmationAnalyzerStepConfig.model_validate(
            _get_default_for_field("multiple_choice_confirmation_analysis")
        )
    )
    # TODO (?): Define Pydantic class for model. Or do we leave it at this?
    # Since we want to unpack the dict as model kwargs.
    models: Dict[str, Dict[str, Any]] = pydantic.Field(
        default_factory=lambda: dict()
    )

    @classmethod
    def from_yaml(
        cls, yaml_path: str | Path
    ) -> "ConfirmationAnalyzerConfig":
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            ConfirmationAnalyzerConfig instance with values from YAML
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)

    # ==helper functions==
    def _step_config(
        self,
        step_config: Optional[ConfirmationAnalyzerStepConfig] = None,
        step_name: Optional[str] = None
    ) -> ConfirmationAnalyzerStepConfig:
        """Internal convenience function."""
        if step_config is None and step_name is None:
            raise ValueError("Either pass a step config or a name of the pipeline step")
        if step_config is None:
            if step_name == "multiple_choice_confirmation_analysis":
                return self.multiple_choice_confirmation_analysis
            elif step_name == "freetext_confirmation_analysis":
                return self.freetext_confirmation_analysis
            else:
                raise ValueError(f"Did not found step config for {step_name}")
        else:
            return step_config

    def get_step_config(
            self,
            step_name: Optional[str] = None,
            step_config: Optional[ConfirmationAnalyzerStepConfig] = None
    ) -> ConfirmationAnalyzerModelStepConfig | MultipleChoiceTaskStepConfig:
        """Get the model specific step config for the given step name.

        The requested `ConfirmationAnalyzerModelStepConfig` is determined
        by either the provided `step_name` or the provided `step_config`.
        If both are given, the `step_config` is used.
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
            step_config: Optional[ConfirmationAnalyzerStepConfig] = None
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
            step_config: Optional[ConfirmationAnalyzerStepConfig] = None
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
            step_config: Optional[ConfirmationAnalyzerStepConfig] = None
    ) -> str:
        """Get the model key for a specific step of the workflow."""
        step_config = self._step_config(step_config, step_name)
        if step_config.used_model_key:
            return step_config.used_model_key
        else:
            return self.used_model_key
