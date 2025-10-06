from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator
)
from typing import Optional
import yaml
from loguru import logger

from evidence_seeker.utils import SubdirConstruction

class UITexts(BaseModel):
    """UI text configuration for a specific language"""

    title: str = "🕵️‍♀️ EvidenceSeeker DemoApp"
    info: Optional[str] = None
    description: str = (
        "Enter a statement in the text field "
        "and have it checked by EvidenceSeeker:"
    )
    statement_label: str = "Statement to check:"
    random_example: str = "Random Example"
    check_statement: str = "Check Statement"
    checking_message: str = (
        "### Checking statement... This could take a few minutes."
    )
    feedback_question: str = "How satisfied are you with the answer?"
    privacy_title: str = "Privacy Notice & Disclaimer"
    warning_label: str = "⚠️ <b>Warning</b>"
    consent_info: str = "**Consent for data processing (Optional)**"
    agree_button: str = "I have taken note of the information"
    password_label: str = "Please enter password for access"
    wrong_password: str = "Wrong password. Please try again."
    continue_text: str = "Continue..."
    server_error: str = "Something went wrong on our end :-("
    disclaimer_text: Optional[str] = None
    data_policy_text: Optional[str] = None
    consent_text: Optional[str] = None


class MultiLanguageUITexts(BaseModel):
    """Multi-language UI text configuration"""

    ui_texts_lang_dict: dict[str, UITexts] = Field(
        default_factory=lambda: {
            "en": UITexts()
        },
    )

    def get_texts(self, language: str) -> UITexts:
        """Get UI texts for a specific language, fallback to English"""
        return self.ui_texts_lang_dict.get(
            language,
            UITexts()
        )


class AppConfig(BaseModel):
    logging: bool = True
    # one of monthly, weekly, yearly, daily, None
    subdirectory_construction: Optional[str] = None
    confirmation_analysis_config_file: str
    preprocessing_config_file: str
    retrieval_config_file: str
    local_base: str | None = None
    result_dir: str
    repo_name: str | None = None
    write_on_github: bool = False
    github_token_name: str = "GITHUB_TOKEN"
    password_protection: bool = False
    password_env_name: str = "EVSE_APP_HASH"
    force_agreement: bool = False
    language: str = "en"
    example_inputs_file: str | None = None
    example_inputs: dict[str, list[str]] | None = {
        "de": [],
        "en": [],
    }
    markdown_template_file: str | None = None
    markdown_template: dict[str, str] | None = None
    save_markdown: bool = True

    translations: dict[str, dict[str, str]] = {
        "en": {}
    }

    # Replace the dict with the Pydantic model
    ui_texts: MultiLanguageUITexts = Field(default_factory=MultiLanguageUITexts)

    # Add a convenience method to get UI texts for current language
    def get_ui_texts(self) -> UITexts:
        """Get UI texts for the current language"""
        logger.debug(f"Getting ui texts for lang: {self.language}")
        return self.ui_texts.get_texts(self.language)

    @computed_field
    @property
    def md_template(self) -> str:
        if self.markdown_template_file is None:
            if self.markdown_template is None:
                raise ValueError("No markdown template or file provided.")
            tmpl = self.markdown_template.get(self.language, None)
            if tmpl is None:
                raise ValueError(
                    "No markdown template available for the specified language."
                )
            return tmpl
        else:
            try:
                with open(self.markdown_template_file, encoding="utf-8") as f:
                    return f.read()
            except Exception:
                raise ValueError("Given 'markdown_template_file' not readable.")

    @computed_field()
    @property
    def examples(self) -> list[str]:
        if self.example_inputs_file is None:
            if self.example_inputs is None:
                raise ValueError("No example inputs or example file provided.")
            example_inputs = self.example_inputs.get(self.language, [])
            return example_inputs
        else:
            try:
                with open(self.example_inputs_file, encoding="utf-8") as f:
                    return f.readlines()
            except Exception:
                raise ValueError("Given 'example_inputs_file' not readable.")

    @staticmethod
    def from_file(file_path: str) -> "AppConfig":
        with open(file_path) as f:
            config = AppConfig(**yaml.safe_load(f))
        return config
    
    @field_validator('subdirectory_construction')
    @classmethod
    def validate_subdirectory_construction(cls, v):
        allowed_values = SubdirConstruction._value2member_map_.keys()
        if (v is not None) and (v not in allowed_values):
            raise ValueError(
                f'subdirectory_construction must be one of {set(allowed_values)}, got {v}'
            )
        return v
