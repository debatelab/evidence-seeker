import pydantic
from typing import Optional
import yaml
import textwrap


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
    password_protection: bool = False
    password_env_name: str = "EVSE_APP_HASH"
    force_agreement: bool = True
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
            "strongly_confirmed": "im hohen Maße bestätigt",
            "confirmed": "bestätigt",
            "weakly_confirmed": "im geringen Maße bestätigt",
            "strongly_disconfirmed": "im hohen Maße widerlegt",
            "disconfirmed": "widerlegt",
            "weakly_disconfirmed": "im geringen Maße widerlegt",
            "inconclusive_confirmation": "weder bestätigt noch widerlegt",
            "The claim is neither confirmed nor disconfirmed.": "Die Aussage wird weder bestätigt noch widerlegt.",
            "The claim is strongly confirmed.": "Die Aussage wird im hohen Maße bestätigt.",
            "The claim is strongly disconfirmed.": "Die Aussage wird im hohen Maße widerlegt.",
            "The claim is weakly confirmed.": "Die Aussage wird in geringem Maße bestätigt.",
            "The claim is weakly disconfirmed.": "Die Aussage wird in geringem Maße widerlegt.",
        },
        "en": {}
    }
    # Multi-language UI texts
    ui_texts: dict[str, dict[str, str]] = {
        "de": {
            "title": "🕵️‍♀️ EvidenceSeeker DemoApp",
            "info": textwrap.dedent("""
                <details>
                <summary>Informationen zur DemoApp</summary>

                + **Grundidee der Pipeline:** Die Pipeline findet in einem ersten Schritt
                unterschiedliche Interpretationen deiner Eingabe und unterscheidet 
                dabei *deskriptive, zuschreibende und normative Aussagen*. Für die gefundenen
                deskriptiven und zuschreibenden Interpretationen wird dann in einer
                *Wissensbasis* nach relevanten Textstellen gesucht und analysiert, inwiefern 
                die Textstellen die gefundene Interpretation bestätigen oder widerlegen.
                Diese Einzelanalysen werden für jede Interpretation in Form eines
                *Bestätigungslevels* aggregiert. Nähere Informationen
                zur Pipeline findest Du [hier](XX).
                + **Verwendete Modelle und Wissensbasis:** In dieser Demo App verwenden wir [XX](#) als Embedding Modell und
                [XX](#) als generatives Sprachmodell. Als Wissensbasis dienen
                alle Ausgaben von "Aus Politik und Zeitgeschichte" aus dem Jahr 2024
                ([Link](https://www.bpb.de/shop/zeitschriften/apuz/?field_filter_thema=all&field_date_content=2024&d=1)).
                + **Beispiele:** Eingaben anderer User:innen und die entsprechenden Ergebnisse der
                Pipeline findest Du unter
                <https://debatelab.github.io/evidence-seeker-results/>.
                
                Die EvidenceSeeker Demoapp ist Teil des vom BMBFSFJ geförderten
                [KIdeKu Projekts](https://compphil2mmae.github.io/research/kideku/).
                Nähere Informationen zur *EvidenceSeeker Boilerplate* findest Du
                [hier](https://debatelab.github.io/evidence-seeker-results).
                </details>
            """).strip(),
            "description": "**Gib eine Aussage in das Textfeld ein und lass sie durch den EvidenceSeeker prüfen:**",
            "statement_label": "Zu prüfende Aussage:",
            "random_example": "Zufälliges Beispiel",
            "check_statement": "Prüfe Aussage",
            "checking_message": "### Aussage wird geprüft... Dies könnte ein paar Minuten dauern.",
            "feedback_question": "Wie zufrieden bist du mit der Antwort?",
            "privacy_title": "Datenschutzhinweis & Disclaimer",
            "consent_info": "**Einwilligung zur Datenweiterverarbeitung (Optional)**",
            "agree_button": "Ich habe die Hinweise zur Kenntnis genommen",
            "password_label": (
                "Die EvidenceSeeker DemoApp ist passwortgeschützt. "
                "Bitte gib für den Zugriff auf die App das Passwort ein."
            ),
            "wrong_password": "Falsches Passwort. Bitte versuche es erneut.",
            "continue": "Weiter...",
            "server_error": "Etwas ist auf unserer Seite schiefgegangen :-(",
            "disclaimer_text": """
                    <div style="background-color:#fff7ed;padding:25px;border-radius: 10px;">
                    <p>⚠️ <b>Disclaimer</b></p>
                    <p>
                    <p>
                    Alle Ausgaben werden von Sprachmodellen generiert und geben nicht
                    zwangsläufig die Einschätzung oder Meinung der Entwickler:innen wieder.
                    </p>
                    <p>
                    Eingegebene Daten werden von Sprachmodellen verarbeitet. Bitte
                    beachte daher, keine personenbezogenen Daten einzugeben - auch weil
                    deine Eingaben unter Umständen gespeichert werden (siehe unten).
                    </p>
                    </p>
                    </div>
                """,
            "data_policy_text": """
                    <div style="background-color:#fff7ed;padding:25px;border-radius: 10px;">
                    <p>🗃️ <b>Datenschutzhinweis</b></p>
                    <p>
                    Auf der Seite <a href="https://debatelab.github.io/evidence-seeker-results/">https://debatelab.github.io/evidence-seeker-results/</a>
                    stellen wir beispielhaft
                    Ergebnisse dar, die von der EvidenceSeeker-Pipeline durch
                    die Interaktion mit Nutzer:innen über diese DemoApp erzeugt wurden.
                    </p>
                    <p>
                    Wir erheben und verwenden <strong>keine personenbezogenen Daten</strong>
                    (sofern sie nicht
                    über das Freitextfeld selbst eingegeben werden) und verwenden
                    <strong>nur</strong> von Nutzer:innen selbst eingegebene
                    Daten sowie den Zeitpunkt der Eingabe, die Rückgabe der Pipeline
                    und etwaiges Feedback durch die Nutzer:innen.
                    </p>
                    <p>
                    Wenn du das EvidenceSeeker Projekt damit unterstützen möchtest,
                    kannst du der Nutzung deiner Eingaben im Folgenden zustimmen.
                    </p>
                    </div>
                """,
            "consent_text": """
                    Ja, meine Anfragen an die EvidenceSeeker Pipeline und deren
                    Ergebnisse dürfen gespeichert und über das
                    EvidenceSeeker Projekt weiter verarbeitet und
                    veröffentlicht werden.
                """,

        },
        "en": {
            # TODO: Check & Revise
            "title": "🕵️‍♀️ EvidenceSeeker DemoApp",
            "description": "Enter a statement in the text field and have it checked by EvidenceSeeker:",
            "statement_label": "Statement to check:",
            "random_example": "Random Example",
            "check_statement": "Check Statement",
            "checking_message": "### Checking statement... This could take a few minutes.",
            "feedback_question": "How satisfied are you with the answer?",
            "privacy_title": "Privacy Notice & Disclaimer",
            "warning_label": "⚠️ <b>Warning</b>",
            "consent_info": "**Consent for data processing (Optional)**",
            "agree_button": "I have taken note of the information",
            "password_label": "Please enter password for access",
            "wrong_password": "Wrong password. Please try again.",
            "continue": "Continue...",
            "server_error": "Something went wrong on our end :-("
        }
    }

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

    @staticmethod
    def from_file(file_path: str) -> "AppConfig":
        with open(file_path) as f:
            config = AppConfig(**yaml.safe_load(f))
        return config
