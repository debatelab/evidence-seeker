import pydantic
from typing import Optional

class AppConfig(pydantic.BaseModel):
    dummy : Optional[bool] = False
    logging : bool = True
    subdirectory_construction : Optional[str] = None
    confirmation_analysis_config_file : str | None = "./config/confirmation_analysis_config.yaml"
    preprocessing_config_file : str | None = "./config/preprocessing_config.yaml"
    retrieval_config_file : str | None = "./config/retrieval_config.yaml"
    local_base : str = "./TMP"
    result_dir : str = "data"
    repo_name : str = "debatelab/evidence-seeker-results"
    github_token: str = "GITHUB_TOKEN"
    example_file : str = "./res/examples.txt"
    result_template_file : str = "./res/result.tmpl"
    translation : dict[str, str] = {
        "ascriptive": "askriptiv",
        "descriptive": "deskriptiv",
        "normative": "normativ",
        "The claim is neither confirmed nor disconfirmed.": "Die Aussage wird weder bestätigt noch widerlegt.",
        "The claim is strongly confirmed.": "Die Aussage wird im hohen Maße bestätigt.",
        "The claim is strongly disconfirmed.": "Die Aussage wird im hohen Maße widerlegt.",
        "The claim is weakly confirmed.": "Die Aussage wird in geringem Maße bestätigt.",
        "The claim is weakly disconfirmed.": "Die Aussage wird in geringem Maße widerlegt."
    }
    warning_text : str = """
        <p>
        Alle Ausgaben werden von Sprachmodellen generiert und geben nicht die Einschätzung oder Meinung der Entwickler:innen wieder. 
        </p>
        <p>
        Eingegebene Daten werden von Sprachmodellen verarbeitet. Bitte beachte daher, keine personenbezogenen Daten einzugeben.
        </p>
    """
    disclaimer_text : str = """
        Auf der Seite [hier einfügen](#) stellen wir beispielhaft Ergebnisse dar, die von der EvidenceSeeker-Pipeline durch die Interaktion mit Nutzer:innen über diese DemoApp erzeugt wurden.

        Wir verwenden **nur** von Nutzer:innen selbst eingegebene Daten, den Zeitpunkt der Eingabe, die Rückgabe der Pipeline und etwaiges Feedback durch die Nutzer:innen.

        Wenn du den Aufbau dieser Seite unterstützen möchtest, kannst du der Nutzung deiner Eingaben im folgenden zustimmen:
    """
    consent_text : str = """
        Ja, meine Anfragen an die EvidenceSeeker-Pipeline und deren Ergebnisse dürfen gespeichert und zur Information über das EvidenceSeeker-Projekt aufbereitet werden.
    """
    
    @pydantic.computed_field
    @property
    def examples(self) -> list[str]:
        try:
            with open(self.example_file, encoding="utf-8") as f:
                return f.readlines()
        except Exception:
            raise ValueError("Given 'example_file' not readable.")