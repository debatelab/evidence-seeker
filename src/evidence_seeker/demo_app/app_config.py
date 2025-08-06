import pydantic
from typing import Optional


class AppConfig(pydantic.BaseModel):
    logging: bool = True
    # one of monthly, weekly, yearly, daily, None
    subdirectory_construction: Optional[str] = None
    confirmation_analysis_config_file: str
    preprocessing_config_file: str
    retrieval_config_file: str
    local_base: str
    result_dir: str
    repo_name: str
    write_on_github: bool = False
    github_token_name: str = "GITHUB_TOKEN"
    language: str = "de"
    example_inputs_file: str | None = None
    example_inputs: dict[str, list[str]] | None = {
        "de": [],
        "en": [],
    }
    markdown_template_file: str | None = None
    markdown_template: dict[str, str] | None = None

    translations: dict[str, dict[str, str]] = {
        "de": {
            "ascriptive": "askriptiv",
            "descriptive": "deskriptiv",
            "normative": "normativ",
            "The claim is neither confirmed nor disconfirmed.": "Die Aussage wird weder bestätigt noch widerlegt.",
            "The claim is strongly confirmed.": "Die Aussage wird im hohen Maße bestätigt.",
            "The claim is strongly disconfirmed.": "Die Aussage wird im hohen Maße widerlegt.",
            "The claim is weakly confirmed.": "Die Aussage wird in geringem Maße bestätigt.",
            "The claim is weakly disconfirmed.": "Die Aussage wird in geringem Maße widerlegt.",
        },
        "en": {}
    }
    warning_text: str = """
        <p>
        Alle Ausgaben werden von Sprachmodellen generiert und geben nicht die Einschätzung oder Meinung der Entwickler:innen wieder. 
        </p>
        <p>
        Eingegebene Daten werden von Sprachmodellen verarbeitet. Bitte beachte daher, keine personenbezogenen Daten einzugeben.
        </p>
    """
    disclaimer_text: str = """
        Auf der Seite [hier einfügen](#) stellen wir beispielhaft Ergebnisse dar, die von der EvidenceSeeker-Pipeline durch die Interaktion mit Nutzer:innen über diese DemoApp erzeugt wurden.

        Wir verwenden **nur** von Nutzer:innen selbst eingegebene Daten, den Zeitpunkt der Eingabe, die Rückgabe der Pipeline und etwaiges Feedback durch die Nutzer:innen.

        Wenn du den Aufbau dieser Seite unterstützen möchtest, kannst du der Nutzung deiner Eingaben im folgenden zustimmen:
    """
    consent_text: str = """
        Ja, meine Anfragen an die EvidenceSeeker-Pipeline und deren Ergebnisse dürfen gespeichert und zur Information über das EvidenceSeeker-Projekt aufbereitet werden.
    """

    @pydantic.computed_field
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



    @pydantic.computed_field
    @property
    def examples(self) -> list[str]:
        if self.example_inputs_file is None:
            if self.example_inputs is None:
                raise ValueError("No example inputs or example file provided.")
            example_inputs = self.example_inputs.get(self.language, [])
            if not example_inputs:
                raise ValueError(
                    "No example inputs available for the specified language."
                )
            return example_inputs
        else:
            try:
                with open(self.example_inputs_file, encoding="utf-8") as f:
                    return f.readlines()
            except Exception:
                raise ValueError("Given 'example_inputs_file' not readable.")
