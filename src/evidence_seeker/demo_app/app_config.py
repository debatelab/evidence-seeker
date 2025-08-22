from pydantic import (
    BaseModel,
    Field,
    computed_field,
)
from typing import Optional
import yaml


class UITexts(BaseModel):
    """UI text configuration for a specific language"""

    title: str
    info: Optional[str] = None
    description: str
    statement_label: str
    random_example: str
    check_statement: str
    checking_message: str
    feedback_question: str
    privacy_title: str
    warning_label: str
    consent_info: str
    agree_button: str
    password_label: str
    wrong_password: str
    continue_text: str
    server_error: str
    disclaimer_text: Optional[str] = None
    data_policy_text: Optional[str] = None
    consent_text: Optional[str] = None


class MultiLanguageUITexts(BaseModel):
    """Multi-language UI text configuration"""

    ui_texts_lang_dict: dict[str, UITexts] = Field(
        default_factory=lambda: {
            "en": UITexts(
                title="🕵️‍♀️ EvidenceSeeker DemoApp",
                description=(
                    "Enter a statement in the text field "
                    "and have it checked by EvidenceSeeker:"
                ),
                statement_label="Statement to check:",
                random_example="Random Example",
                check_statement="Check Statement",
                checking_message=(
                    "### Checking statement... This could take a few minutes."
                ),
                feedback_question="How satisfied are you with the answer?",
                privacy_title="Privacy Notice & Disclaimer",
                warning_label="⚠️ <b>Warning</b>",
                consent_info="**Consent for data processing (Optional)**",
                agree_button="I have taken note of the information",
                password_label="Please enter password for access",
                wrong_password="Wrong password. Please try again.",
                continue_text="Continue...",
                server_error="Something went wrong on our end :-(",
            )
        },
    )

    def get_texts(self, language: str) -> UITexts:
        """Get UI texts for a specific language, fallback to English"""
        return self.ui_texts_lang_dict.get(
            language,
            # Fallback to English for unsupported languages
            self.ui_texts_lang_dict["en"]
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
    # TODO: refactor when we have a better solution in util.results_as_markdown
    group_docs_by_sources: bool = False
    save_markdown: bool = True

    translations: dict[str, dict[str, str]] = {
        "en": {}
    }

    # Replace the dict with the Pydantic model
    ui_texts: MultiLanguageUITexts = Field(default_factory=MultiLanguageUITexts)

    # Add a convenience method to get UI texts for current language
    def get_ui_texts(self) -> UITexts:
        """Get UI texts for the current language"""
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
