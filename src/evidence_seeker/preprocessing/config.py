"PreprocessingConfig"

from typing import Any, Dict, List

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
    timeout: int = 120
    verbose: bool = False
    used_model_key: str = "model_1"
    freetext_descriptive_analysis: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="freetext_descriptive_analysis",
            description="Instruct the assistant to carry out free-text factual/descriptive analysis.",
            prompt_template=(
                "The following claim has been submitted for fact-checking.\n\n"
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
                "End your analysis with a short list of all identified factual or descriptive statements. "
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
                "The following claim has been submitted for analysing its descriptive content.\n"
                "<claim>{claim}</claim>\n"
                "The analysis yielded the following results:\n"
                "<results>\n"
                "{descriptive_analysis}\n"
                "</results>\n"
                "Building on this analysis:\n"
                "- I want you to list all identified factual or descriptive statements. Only include "
                "clear cases, i.e. statements that are unambiguously factual or descriptive.\n"
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
                "The following claim has been submitted for fact-checking.\n\n"
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
                "End your analysis with a short list including all identified ascriptions. "
                "Formulate each ascription as a concise statement, such that it is transparent to "
                "whom it is attributed."
            ),
        )
    )
    list_ascriptive_statements: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
            name="list_ascriptive_statements",
            description="Instruct the assistant to list ascriptions.",
            prompt_template=(
                "The following claim has been submitted for ascriptive content analysis.\n"
                "<claim>{claim}</claim>\n"
                "The analysis yielded the following results:\n"
                "<results>\n"
                "{ascriptive_analysis}\n"
                "</results>\n"
                "Based on this analysis:\n"
                "- List all identified ascriptions. Clearly state each ascription as a concise "
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
                "The following claim has been submitted for fact-checking.\n\n"
                "<claim>{claim}</claim>\n\n"
                "Before we proceed with retrieving evidence items, we carefully analyse the claim. "
                "Your task is to contribute to this preparatory analysis, as detailed below.\n"
                "In particular, you should thoroughly discuss whether the claim contains or implies "
                "normative statements, such as value judgements, recommendations, or evaluations. "
                "If so, try to identify them and render them in your own words.\n"
                "In doing so, watch out for ambiguity and vagueness in the claim. Make alternative "
                "interpretations explicit.\n"
                "However, avoid reading normative content into the claim without textual evidence.\n"
                "End your analysis with a short list of all identified normative statements. "
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
                "The following claim has been submitted for normative content analysis.\n"
                "<claim>{claim}</claim>\n"
                "The analysis yielded the following results:\n"
                "<results>\n"
                "{normative_analysis}\n"
                "</results>\n"
                "Based on this analysis:\n"
                "- List all identified normative statements (e.g., value judgements, "
                "recommendations, or evaluations).\n"
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
                "Please generate a single sentence that clearly states the negation.\n"
                "<statement>\n"
                "{statement}\n"
                "</statement>\n"
                "Provide only the negated statement without any additional comments."
            ),
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
        }
    )
