"PreprocessingConfig"

from typing import Any, Dict, Optional

import pydantic
from loguru import logger
from llama_index.core import ChatPromptTemplate

from evidence_seeker.backend import GuidanceType


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
    config_version: str = "v0.1"
    description: str = "Configuration of EvidenceSeeker's preprocessing component."
    system_prompt: str = (
        "You are a helpful assistant with outstanding expertise in critical thinking and logico-semantic analysis. \n"
        "You have a background in philosophy and experience in fact checking and debate analysis.\n"
        "You read instructions carefully and follow them precisely. You give concise and clear answers."
    )
    language: str = "DE"
    timeout: int = 900
    # Whether or not the workflow/pipeline should print additional informative messages
    # during execution.
    verbose: bool = False
    env_file: str | None = None

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
        default_factory=lambda: PreprocessorStepConfig(
            name="freetext_descriptive_analysis",
            description="Instruct the assistant to carry out free-text factual/descriptive analysis.",
            llm_specific_configs={
                "default": PreprocessorModelStepConfig(
                    prompt_template=(
                        "The following {language} claim has been submitted for fact-checking.\n\n"
                        "<claim>{claim}</claim>\n\n"
                        "Before we proceed with retrieving evidence items, we carefully analyse the claim. "
                        "Your task is to contribute to this preparatory analysis, as detailed below.\n"
                        "In particular, you should thoroughly discuss whether the claim contains or implies "
                        "factual or descriptive statements, which can be verified or falsified by empirical "
                        "observation or through scientific analysis, and which may include, for example, "
                        "descriptive reports, historical facts, or scientific claims.\n"
                        "If so, try to identify them and render them in your own words.\n"
                        "In doing so, watch out for ambiguity and vagueness in the claim. Make alternative "
                        "interpretations explicit.\n"
                        "End your analysis with a short list of all identified factual or descriptive statements in {language}. "
                        "Formulate each statement in a concise manner and such that its factual nature stands "
                        "out clearly."
                    ),
                )
            }
        )
    )
    list_descriptive_statements: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig(
            name="list_descriptive_statements",
            description="Instruct the assistant to list factual claims.",
            llm_specific_configs={
                "default": PreprocessorModelStepConfig(
                    prompt_template=(
                        "We have previously analysed the descriptive content of the following {language} claim:\n"
                        "<claim>{claim}</claim>\n"
                        "The analysis yielded the following results:\n\n"
                        "<results>\n"
                        "{descriptive_analysis}\n"
                        "</results>\n\n"
                        "Your task is to list all factual or descriptive {language} statements identified "
                        "in the previous analysis. Only include clear cases, i.e. statements that are unambiguously "
                        "factual or descriptive.\n"
                        "If you did not identify any descriptive statements, make sure to return an empty list containing no strings, not even empty ones.\n"
                        "Format your (possibly empty) list of statements as a JSON object.\n"
                        "Do not include any other text than the JSON object."
                    ),
                    guidance_type=GuidanceType.PYDANTIC.value
                )
            },
        ),
    )
    freetext_ascriptive_analysis: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig(
            name="freetext_ascriptive_analysis",
            description="Instruct the assistant to carry out free-text ascriptions analysis.",
            llm_specific_configs={
                "default": PreprocessorModelStepConfig(
                    prompt_template=(
                        "The following {language} claim has been submitted for fact-checking.\n\n"
                        "<claim>{claim}</claim>\n\n"
                        "Before we proceed with retrieving evidence items, we carefully analyse the claim. "
                        "Your task is to contribute to this preparatory analysis, as detailed below.\n"
                        "In particular, you should thoroughly discuss whether the claim makes any explicit "
                        "ascriptions, that is, whether it explicitly ascribes a statement to a person or an "
                        "organisation (e.g., as something the person has said, believes, acts on etc.) "
                        "rather than plainly asserting that statement straightaway.\n"
                        "If so, clarify which statements are ascribed to whom exactly and in which ways.\n"
                        "In doing so, watch out for ambiguity and vagueness in the claim. Make alternative "
                        "interpretations explicit.\n"
                        "Conclude your analysis with a short list of all identified ascriptions: "
                        "Formulate each statement in a concise manner, and such that it is transparent to "
                        "whom it is attributed. Render the clarified ascriptions in {language}."
                    ),
                )
            },
        )
    )
    list_ascriptive_statements: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig(
            name="list_ascriptive_statements",
            description="Instruct the assistant to list ascriptions.",
            llm_specific_configs={
                "default": PreprocessorModelStepConfig(
                    prompt_template=(
                        "The following {language} claim has been submitted for ascriptive content analysis.\n"
                        "<claim>{claim}</claim>\n"
                        "The analysis yielded the following results:\n\n"
                        "<results>\n"
                        "{ascriptive_analysis}\n"
                        "</results>\n\n"
                        "Your task is to list all ascriptions identified in this analysis. "
                        "Clearly state each ascription as a concise {language} "
                        "statement, such that it is transparent to whom it is attributed. Only include "
                        "ascriptions that are explicitly attributed to a specific person or organisation.\n"
                        "If you did not identify any ascriptions, make sure to return an empty list containing no strings, not even empty ones..\n"
                        "Format your (possibly empty) list of ascriptions as a JSON object.\n"
                        "Do not include any other text than the JSON object."
                    ),
                    guidance_type=GuidanceType.PYDANTIC.value
                )
            },
        ),
    )
    freetext_normative_analysis: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig(
            name="freetext_normative_analysis",
            description="Instruct the assistant to carry out free-text normative analysis.",
            llm_specific_configs={
                "default": PreprocessorModelStepConfig(
                    prompt_template=(
                        "The following {language} claim has been submitted for fact-checking.\n\n"
                        "<claim>{claim}</claim>\n\n"
                        "Before we proceed with retrieving evidence items, we carefully analyse the claim. "
                        "Your task is to contribute to this preparatory analysis, as detailed below.\n"
                        "In particular, you should thoroughly discuss whether the claim contains or implies "
                        "normative statements, such as value judgements, recommendations, or evaluations. "
                        "If so, try to identify them and render them in your own words.\n"
                        "In doing so, watch out for ambiguity and vagueness in the claim. Make alternative "
                        "interpretations explicit. "
                        "However, avoid reading normative content into the claim without textual evidence.\n\n"
                        "End your analysis with a short list of all identified normative statements in {language}. "
                        "Formulate each statement in a concise manner and such that its normative nature "
                        "stands out clearly."
                    ),
                )
            },
        ),
    )
    list_normative_statements: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig(
            name="list_normative_statements",
            description="Instruct the assistant to list normative claims.",
            llm_specific_configs={
                "default": PreprocessorModelStepConfig(
                    prompt_template=(
                        "The following {language} claim has been submitted for normative content analysis.\n"
                        "<claim>{claim}</claim>\n"
                        "The analysis yielded the following results:\n\n"
                        "<results>\n"
                        "{normative_analysis}\n"
                        "</results>\n\n"
                        "Your task is to list all normative statements identified in this analysis "
                        "(e.g., value judgements, recommendations, or evaluations) in {language}.\n"
                        "If you did not identify any normative statements, make sure to return an empty list containing no strings, not even empty ones.\n"
                        "Format your (possibly empty) list of statements as a JSON object.\n"
                        "Do not include any other text than the JSON object."
                    ),
                    guidance_type=GuidanceType.PYDANTIC.value
                )
            },
        ),
    )
    negate_claim: PreprocessorStepConfig = pydantic.Field(
        default_factory=lambda: PreprocessorStepConfig(
            name="negate_claim",
            description="Instruct the assistant to negate a claim.",
            llm_specific_configs={
                "default": PreprocessorModelStepConfig(
                    prompt_template=(
                        "Your task is to express the opposite of the following statement in plain "
                        "and unequivocal language.\n"
                        "Please generate a single {language} sentence that clearly states the negation.\n"
                        "<statement>\n"
                        "{statement}\n"
                        "</statement>\n"
                        "Provide only the negated statement in {language} without any additional comments."
                    ),
                )
            },
        ),
    )
    models: Dict[str, Dict[str, Any]] = pydantic.Field(
        default_factory=lambda: dict()
    )

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
