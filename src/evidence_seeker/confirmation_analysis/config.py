"confirmation_analysis.py"

from typing import Any, Dict, List

import pydantic


class PipelineStepConfig(pydantic.BaseModel):
    name: str
    description: str
    prompt_template: str
    used_model_key: str | None = None
    system_prompt: str | None = None
    options: List[str] | None = None
    claim_option: str | None = None
    regex_str: str | None = None


class ConfirmationAnalyzerConfig(pydantic.BaseModel):
    config_version: str = "v0.1"
    description: str = "Erste Version einer Konfiguration für den ConfirmationAnalyzerConfig der EvidenceSeeker Boilerplate."
    system_prompt: str = (
        "You are a helpful assistant with outstanding expertise in critical thinking and logico-semantic analysis. "
        "You have a background in philosophy and experience in fact checking and debate analysis.\n"
        "You read instructions carefully and follow them precisely. You give concise and clear answers."
    )
    timeout: int = 60
    verbose: bool = False
    used_model_key: str = "model_1"
    freetext_confirmation_analysis: PipelineStepConfig = pydantic.Field(
        default_factory=lambda: PipelineStepConfig(
                name="freetext_confirmation_analysis",
                description="Instruct the assistant to carry out free-text RTE analysis.",
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
    )
    multiple_choice_confirmation_analysis: PipelineStepConfig = pydantic.Field(
         default_factory=lambda: PipelineStepConfig(
                name="multiple_choice_confirmation_analysis",
                used_model_key="model_3",
                description="Multiple choice RTE task given CoT trace.",
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
                    "(A) Entailment: The TEXT provides sufficient evidence to support the HYPOTHESIS.\n"
                    "(B) Contradiction: The TEXT provides evidence that contradicts the HYPOTHESIS.\n"
                    "(C) Neutral: The TEXT neither supports nor contradicts the HYPOTHESIS."
                ),
                options=["(A", "(B", "(C"],
                claim_option="(A",
                regex_str=r"(\(A)|(\(B)|(\(C)",
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
