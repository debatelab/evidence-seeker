"confirmation_analysis.py"

from typing import Any, Dict, List, Optional, Self
from loguru import logger
from llama_index.core import ChatPromptTemplate

import re
import pydantic
import enum

from evidence_seeker.backend import GuidanceType

class LogProbsType(enum.Enum):
    OPENAI_LIKE = "openai_like"
    ESTIMATE = "estimate"

def json_schema(pattern : str) -> Dict[str, Any]:
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

class PipelineModelStepConfig(pydantic.BaseModel):
    prompt_template: str
    system_prompt: str | None = None
    # following fields are only used for multiple choice tasks
    answer_labels: Optional[List[str]] = None
    claim_option: Optional[str] = None
    n_repetitions_mcq: int = 1
    # ToDo: As set of strings
    answer_options: Optional[List[str]] = None
    delim_str: Optional[str] = "."
    # Fields used for constrained decoding
    guidance_type: Optional[str] = GuidanceType.REGEX.value
    constrained_decoding_regex: Optional[str] = None
    constrained_decoding_grammar: Optional[str] = None
    # used for regex-based validation of the output for `GuidanceType.PROMPTED`
    validation_regex: Optional[str] = None
    # log probs
    logprobs_type: Optional[str] = LogProbsType.OPENAI_LIKE.value
    # JSON schema for JSON Guidance
    @pydantic.computed_field
    @property
    def json_schema(self) -> Optional[str|Dict[str, Any]]:
        if self.guidance_type == GuidanceType.JSON.value and self.answer_labels is not None and len(self.answer_labels)!=0:
            return json_schema(pattern=rf"^({'|'.join(self.answer_labels)})$")
        return None
    # Validation of answer labels for JSON Guidance
    @pydantic.model_validator(mode='after')
    def check_answer_labels(self) -> Self:
        if self.guidance_type == GuidanceType.JSON.value:
            if self.answer_labels is None or len(self.answer_labels)==0:
                raise ValueError('Please provide possible answer labels for multiple choice tasks when using JSON Guidance.')
            else:
                seen = set()
                seen_twice = set(x for x in self.answer_labels if x in seen or seen.add(x))
                valid_labels = True; [valid_labels := valid_labels & (re.match(r"^[a-zA-Z0-9]$", l) is not None) for l in self.answer_labels]
                if len(seen_twice) != 0 or not valid_labels:
                    raise ValueError("JSON Guidance assumes unique and single characters as answer labels. Possible characters are ASCII characters in the ranges of a-z, A-Z and 0-9.")
        return self

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
    timeout: int = 3000
    verbose: bool = False
    #used_model_key: Optional[str] = "lmstudio"
    used_model_key: Optional[str] = "together.ai"
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
                    )
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
                    "lmstudio": PipelineModelStepConfig(
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
                            "Please sum up this result by deciding which of the following choices is correct.\n"
                            "\n"
                            "{answer_options}\n"
                            "\n"
                            "Just answer with the label ('A', 'B' or 'C') of the correct choice.\n"
                            "\n"
                        ),
                        answer_labels=["A", "B", "C"],
                        n_repetitions_mcq=3,
                        claim_option="Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.",
                        delim_str=".",
                        answer_options=[
                            "Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.",
                            "Contradiction: The TEXT provides evidence that contradicts the HYPOTHESIS.",
                            "Neutral: The TEXT neither supports nor contradicts the HYPOTHESIS.",
                        ],
                        guidance_type=GuidanceType.PROMPTED.value,
                        logprobs_type=LogProbsType.ESTIMATE.value,
                        validation_regex=r"^[\.\(]?(A|B|C)[\.\):]?$",
                    ),
                    "together.ai": PipelineModelStepConfig(
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
                            "Please sum up this result by deciding which of the following choices is correct.\n"
                            "\n"
                            "{answer_options}\n"
                            "\n"
                            "Just answer with the label ('A', 'B' or 'C') of the correct choice.\n"
                            "\n"
                        ),
                        answer_labels=["A", "B", "C"],
                        claim_option="Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.",
                        delim_str="",
                        answer_options=[
                            "Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.",
                            "Contradiction: The TEXT provides evidence that contradicts the HYPOTHESIS.",
                            "Neutral: The TEXT neither supports nor contradicts the HYPOTHESIS.",
                        ],
                        guidance_type=GuidanceType.JSON.value, # ='json'
                        logprobs_type=LogProbsType.OPENAI_LIKE.value, # ='openai_like'
                        n_repetitions_mcq=3,
                        #guidance_type=GuidanceType.PROMPTED.value,
                        #logprobs_type=LogProbsType.ESTIMATE.value,
                        validation_regex=r"^[\.\(]?(A|B|C)[\.\):]?$",
                        #json_schema=Answer.model_json_schema()
                    ),
                    "hf_inference_api": PipelineModelStepConfig(
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
                            "Please sum up this result by deciding which of the following choices is correct.\n"
                            "\n"
                            "{answer_options}\n"
                            "\n"
                            "Just answer with the label ('A', 'B' or 'C') of the correct choice.\n"
                            "\n"
                        ),
                        answer_labels=["A", "B", "C"],
                        claim_option="Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.",
                        delim_str=".",
                        answer_options=[
                            "Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.",
                            "Contradiction: The TEXT provides evidence that contradicts the HYPOTHESIS.",
                            "Neutral: The TEXT neither supports nor contradicts the HYPOTHESIS.",
                        ],
                        n_repetitions_mcq=1,
                        guidance_type=GuidanceType.REGEX.value,
                        constrained_decoding_regex="(A)|(B)|(C)",
                        logprobs_type=LogProbsType.OPENAI_LIKE.value,
                    ),
                }
            ),
    )
    # TODO (?): Define Pydantic class for model. Or do we leave it at this?
    # Since we want to unpack the dict as model kwargs.
    models: Dict[str, Dict[str, Any]] = pydantic.Field(
        default_factory=lambda: {
            "model_1": {
                "name": "Mistral-7B-Instruct-v0.2",
                "description": "HF inference API",
                "base_url": "https://api-inference.huggingface.co/v1/",
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "api_key_name": "HF_TOKEN_EVIDENCE_SEEKER",
                "backend_type": "openai",
                "max_tokens": 1024,
                "temperature": 0.2,
            },
            'lmstudio': {
                "name": "llama-3.2-1b-instruct",
                "description": "Local model served via LMStudio",
                "base_url": "http://127.0.0.1:1234/v1/",
                "model": "llama-3.2-1b-instruct",
                "backend_type": "openai",
                "max_tokens": 1024,
                "temperature": 0.2,
                "api_key": "not_needed",
                "timeout": 260
            },
            'together.ai': {
                "name": "Meta-Llama-3-Instruct",
                "description": "Model served via Together.ai over HuggingFace",
                "base_url": "https://router.huggingface.co/together/v1",
                "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                "api_key_name": "hf_debatelab_inference_provider",
                "backend_type": "openai",
                "default_headers": {"X-HF-Bill-To": "DebateLabKIT"},
                "max_tokens": 1024,
                "temperature": 0.2,
                "timeout": 260
            },
            'hf_inference_api': {
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "description": "Model served over HuggingFace Inference API",
                "base_url": "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.1-8B-Instruct/v1",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "api_key_name": "hf_debatelab_inference_provider",
                "backend_type": "tgi",
                "default_headers": {"X-HF-Bill-To": "DebateLabKIT"},
                "max_tokens": 1024,
                "temperature": 0.2,
                "timeout": 260
            },
        }
    )

    # ==helper functions==
    def _step_config(
        self,
        step_config: Optional[PipelineStepConfig] = None,
        step_name: Optional[str] = None
    ) -> PipelineStepConfig:
        """Internal convenience function."""
        if step_config is None and step_name is None:
            raise ValueError("Either pass a step config of a name of the pipeline step")
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
            step_config: Optional[PipelineStepConfig] = None
    ) -> PipelineModelStepConfig:
        """Get the model specific step config for the given step name."""
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
            step_config: Optional[PipelineStepConfig] = None
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
            step_config: Optional[PipelineStepConfig] = None
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
            step_config: Optional[PipelineStepConfig] = None
    ) -> str:
        """Get the model key for a specific step of the workflow."""
        step_config = self._step_config(step_config, step_name)
        if step_config.used_model_key:
            return step_config.used_model_key
        else:
            return self.used_model_key
