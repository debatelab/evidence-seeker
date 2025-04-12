"confirmation_analysis.py"

from typing import Any, Dict, List, Optional
from loguru import logger
from llama_index.core import ChatPromptTemplate

import pydantic


class PipelineModelStepConfig(pydantic.BaseModel):
    prompt_template: str
    system_prompt: str | None = None
    # following fields are only used for multiple choice tasks
    answer_labels: List[str] | None = None
    claim_option: str | None = None
    # ToDo: As set of strings
    answer_options: List[str] = None
    delim_str: str | None = "."
    constrained_decoding_regex: Optional[str] = None


class PipelineStepConfig(pydantic.BaseModel):
    name: str
    description: str
    used_model_key: str | None = None
    llm_specific_configs: Dict[str, PipelineModelStepConfig]


class ConfirmationAnalyzerConfig(pydantic.BaseModel):
    config_version: str = "v0.2"
    description: str = "Erste Version einer Konfiguration für den ConfirmationAnalyzerConfig der EvidenceSeeker Boilerplate."
    system_prompt: str = (
        "You are a helpful assistant with outstanding expertise in critical thinking and logico-semantic analysis. "
        "You have a background in philosophy and experience in fact checking and debate analysis.\n"
        "You read instructions carefully and follow them precisely. You give concise and clear answers."
    )
    timeout: int = 240
    verbose: bool = False
    used_model_key: Optional[str] = None
    freetext_confirmation_analysis: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="freetext_confirmation_analysis",
            description="Instruct the assistant to carry out free-text RTE analysis.",
            llm_specific_configs={
                "default": PipelineModelStepConfig(
                    prompt_template=(
                        "Determine the relationship between the following two texts:\n"
                        "<TEXT>{evidence_item}</TEXT>\n"
                        "\n"
                        "<HYPOTHESIS>{statement}</HYPOTHESIS>\n"
                        "Does the TEXT entail, contradict, or neither entail nor contradict the HYPOTHESIS?\n"
                        "Classify the relationship as one of the following:\n"
                        "Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.\n"
                        "Contradiction: The TEXT provides evidence that contradicts the HYPOTHESIS.\n"
                        "Neutral: The TEXT neither supports nor contradicts the HYPOTHESIS.\n"
                        "Please discuss this question thoroughly before providing your final answer."
                    ),
                )
            }
        )
    )
    multiple_choice_confirmation_analysis: PipelineStepConfig = pydantic.Field(
         default_factory=lambda: PipelineStepConfig(
                name="multiple_choice_confirmation_analysis",
                description="Multiple choice RTE task given CoT trace.",
                llm_specific_configs={
                    "default": PipelineModelStepConfig(
                        prompt_template=(
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
                        ),
                        answer_labels=["(A", "(B", "(C"],
                        claim_option="Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.",
                        delim_str=")",
                        answer_options=[
                            "Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.",
                            "Contradiction: The TEXT provides evidence that contradicts the HYPOTHESIS.",
                            "Neutral: The TEXT neither supports nor contradicts the HYPOTHESIS.",
                        ],
                        constrained_decoding_regex=r"^(\(A|\(B|\(C)$"
                    ),
                }
            ),
    )
    models: Dict[str, Dict[str, Any]] = pydantic.Field(
        default_factory=lambda: {
            "model_1": {
                "name": "Llama-3.1-70B-Instruct",
                "description": "NVIDEA NIM API (kostenpflichtig über DebateLab Account)",
                "base_url": "https://huggingface.co/api/integrations/dgx/v1",
                "model": "meta-llama/Llama-3.1-70B-Instruct",
                "api_key_name": "HF_TOKEN_EVIDENCE_SEEKER",
                "backend_type": "nim",
                "max_tokens": 2048,
                "temperature": 0.2,
            },
            "model_2": {
                "name": "Mistral-7B-Instruct-v0.2",
                "description": "HF inference API",
                "base_url": "https://api-inference.huggingface.co/v1/",
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "api_key_name": "HF_TOKEN_EVIDENCE_SEEKER",
                "backend_type": "openai",
                "max_tokens": 1024,
                "temperature": 0.2,
            },
            "model_3": {
                "name": "Llama-3.2-3B-Instruct",
                "description": "HF dedicated endpoint (debatelab)",
                "base_url": "https://dchi8b9swca6gxbe.eu-west-1.aws.endpoints.huggingface.cloud/v1/",
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "api_key_name": "HF_TOKEN_EVIDENCE_SEEKER",
                "backend_type": "tgi",
                "max_tokens": 2048,
                "temperature": 0.2,
            },
            "model_4": {
                "name": "Spätzle 8B",
                "description": "Kriton@DebateLab",
                "base_url": "http://kriton.philosophie.kit.edu:8080/v1/",
                "model": "tgi",
                "api_key": "no-key-required",
                "backend_type": "tgi",
                "max_tokens": 2048,
                "temperature": 0.2,
            },
            "model_5": {
                "name": "Llama-3.1-70B-Instruct",
                "description": "HF dedicated endpoint (debatelab)",
                "base_url": "https://ev6086dt6s7nn1b5.us-east-1.aws.endpoints.huggingface.cloud/v1/",
                "model": "meta-llama/Llama-3.1-70B-Instruct",
                "api_key_name": "HF_TOKEN_EVIDENCE_SEEKER",
                "backend_type": "tgi",
                "max_tokens": 2048,
                "temperature": 0.2,
            },
        }
    )

    # ==helper functions==

    def get_step_config(
            self,
            step_config: PipelineStepConfig
    ) -> PipelineModelStepConfig:
        """Get the model specific step config for the given step name."""

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
                logger.error(
                    f"Default step config for {step_config.name} "
                    "not found in config."
                )
                raise ValueError(
                    f"Default step config for {step_config.name} "
                    "not found in config."
                )
            model_specific_conf = step_config.llm_specific_configs["default"]
        return model_specific_conf

    def get_chat_template(
            self, step_config: PipelineStepConfig
    ) -> ChatPromptTemplate:

        model_specific_conf = self.get_step_config(step_config)
        prompt_template = model_specific_conf.prompt_template

        return ChatPromptTemplate.from_messages(
            [
                ("system", self.get_system_prompt(step_config)),
                ("user", prompt_template),
            ]
        )

    def get_system_prompt(self, step_config: PipelineStepConfig) -> str:
        """Get the system prompt for a specific step of the workflow."""
        model_specific_conf = self.get_step_config(step_config)
        if model_specific_conf.system_prompt:
            return model_specific_conf.system_prompt
        else:
            return self.system_prompt

    def get_model_key(self, step_config: PipelineStepConfig) -> str:
        """Get the model key for a specific step of the workflow."""
        if step_config.used_model_key:
            return step_config.used_model_key
        else:
            return self.used_model_key
