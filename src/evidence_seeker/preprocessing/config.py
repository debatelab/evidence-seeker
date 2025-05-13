"PreprocessingConfig"

from typing import Any, Dict

import pydantic


class PipelineStepConfig(pydantic.BaseModel):
    name: str
    description: str
    prompt_template: str
    used_model_key: str | None = None
    system_prompt: str | None = None


class ClaimPreprocessingConfig(pydantic.BaseModel):
    config_version: str = "v0.1"
    description: str = "Erste Version einer Konfiguration für den Preprocessor der EvidenceSeeker Boilerplate."
    system_prompt: str = (
        "You are a helpful assistant with outstanding expertise in critical thinking and logico-semantic analysis. "
        "You have a background in philosophy and experience in fact checking and debate analysis.\n"
        "You read instructions carefully and follow them precisely. You give concise and clear answers."
    )
    language: str = "DE"
    timeout: int = 120
    verbose: bool = False
    used_model_key: str = "together.ai"
    freetext_descriptive_analysis: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="freetext_descriptive_analysis",
            description="Instruct the assistant to carry out free-text factual/descriptive analysis.",
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
    )
    list_descriptive_statements: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="list_descriptive_statements",
            description="Instruct the assistant to list factual claims.",
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
                "Format your (possibly empty) list of statements as a JSON object.\n"
                "Do not include any other text than the JSON object."
            ),
        ),
    )
    freetext_ascriptive_analysis: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="freetext_ascriptive_analysis",
            description="Instruct the assistant to carry out free-text ascriptions analysis.",
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
    )
    list_ascriptive_statements: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="list_ascriptive_statements",
            description="Instruct the assistant to list ascriptions.",
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
                "Format your (possibly empty) list of statements as a JSON object.\n"
                "Do not include any other text than the JSON object."
            ),
        ),
    )
    freetext_normative_analysis: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="freetext_normative_analysis",
            description="Instruct the assistant to carry out free-text normative analysis.",
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
        ),
    )
    list_normative_statements: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="list_normative_statements",
            description="Instruct the assistant to list normative claims.",
            prompt_template=(
                "The following {language} claim has been submitted for normative content analysis.\n"
                "<claim>{claim}</claim>\n"
                "The analysis yielded the following results:\n\n"
                "<results>\n"
                "{normative_analysis}\n"
                "</results>\n\n"
                "Your task is to list all normative statements identified in this analysis "
                "(e.g., value judgements, recommendations, or evaluations) in {language}.\n"
                "Format your (possibly empty) list of statements as a JSON object.\n"
                "Do not include any other text than the JSON object."
            ),
        ),
    )
    negate_claim: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="negate_claim",
            description="Instruct the assistant to negate a claim.",
            prompt_template=(
                "Your task is to express the opposite of the following statement in plain "
                "and unequivocal language.\n"
                "Please generate a single {language} sentence that clearly states the negation.\n"
                "<statement>\n"
                "{statement}\n"
                "</statement>\n"
                "Provide only the negated statement in {language} without any additional comments."
            ),
        ),
    )
    models: Dict[str, Dict[str, Any]] = pydantic.Field(
        default_factory=lambda: {
            'lmstudio': {
                "name": "meta-llama-3.1-8b-instruct",
                # "name": "llama-3.2-1b-instruct",
                "description": "Local model served via LMStudio",
                "base_url": "http://127.0.0.1:1234/v1/",
                "model": "meta-llama-3.1-8b-instruct",
                # "model": "llama-3.2-1b-instruct",
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
        }
    )
