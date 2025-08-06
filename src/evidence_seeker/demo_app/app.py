from __future__ import annotations

import gradio as gr
import random
import dotenv
import os
from loguru import logger
from datetime import datetime, timezone
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

from evidence_seeker.demo_app.app_config import AppConfig

from evidence_seeker import (
    EvidenceSeeker,
    EvidenceSeekerResult,
    result_as_markdown,
    log_result
)

from evidence_seeker import (
    CheckedClaim,
    Document,
    StatementType
)

# ### APP CONFIGURATION ####
dotenv.load_dotenv()

config_file_path = os.getenv("APP_CONFIG_FILE", None)

if config_file_path is None:
    raise ValueError("Missing configruation file.")
APP_CONFIG = AppConfig.from_file(config_file_path)

UI_TEST_MODE = os.getenv("UI_TEST_MODE", False)

# used for UI_TEST_MODE
_dummy_docs = [
    Document(
        text='While there is high confidence that oxygen levels have ...',
        uid='1f47ce98-4105-4ddc-98a9-c4956dab2000',
        metadata={
            'page_label': '74',
            'file_name': 'IPCC_AR6_WGI_TS.pdf',
            'author': 'IPCC Working Group I',
            'original_text': 'While there is low confidence in 20th century ...',
            'url': 'www.dummy_url.com',
            'title': 'Dummy Title'
        }
    ),
    Document(
        text='Based on recent refined \nanalyses of the ... ',
        uid='6fcd6c0f-99a1-48e7-881f-f79758c54769',
        metadata={
            'page_label': '74',
            'file_name': 'IPCC_AR6_WGI_TS.pdf',
            'author': 'IPCC Working Group I',
            'original_text': 'The AMOC was relatively stable during the past ...',
            'url': 'www.dummy_url.com',
            'title': 'Dummy Title'
        }
    ),
]

_dummy_claims = [
    CheckedClaim(
        text="The AMOC is slowing down",
        negation="The AMOC is not changing",
        uid="123",
        documents=_dummy_docs,
        n_evidence=2,
        statement_type=StatementType.DESCRIPTIVE,
        average_confirmation=0.2,
        evidential_uncertainty=0.1,
        verbalized_confirmation="confirmed...",
        confirmation_by_document={
            "1f47ce98-4105-4ddc-98a9-c4956dab2000": 0.1,
            "6fcd6c0f-99a1-48e7-881f-f79758c54769": 0.3,
        },
    ),
]


def check_password(input_password: str, hash: str) -> bool:
    ph = PasswordHasher()
    try:
        ph.verify(hash, input_password)
        return True
    except VerificationError:
        return False


def auth(pw: str, password_authenticated: bool):
    output = ""
    b = gr.Textbox(value="")
    if APP_CONFIG.password_env_name not in os.environ:
        output = "Etwas ist auf unserer Seite schiefgegangen :-("
    elif not check_password(pw, os.environ[APP_CONFIG.password_env_name]):
        output = "Falsches Passwort. Bitte versuche es erneut."
    else:
        output = "Weiter..."
        password_authenticated = True
    return output, password_authenticated, b


def reactivate(check_btn, statement):
    if statement.strip() != "":
        check_btn = gr.Button(visible=True, interactive=True)
    good = gr.Button(visible=True, interactive=True)
    bad = gr.Button(visible=True, interactive=True)
    feedback = gr.Markdown(visible=True)
    return feedback, check_btn, good, bad


def deactivate():
    check_btn = gr.Button(interactive=False)
    good = gr.Button(interactive=False, variant="secondary", visible=False)
    bad = gr.Button(interactive=False, variant="secondary", visible=False)
    feedback = gr.Markdown(visible=False)
    return check_btn, good, bad, feedback


def log_feedback(clicked_button, last_result: EvidenceSeekerResult):
    choice = "positive" if clicked_button == "👍" else "negative"
    last_result.feedback["binary"] = (
        None if last_result.feedback["binary"] == choice else choice
    )
    logger.log("INFO", f"{last_result.feedback['binary']} feedback on results")
    return gr.Button(
        variant="primary" if last_result.feedback["binary"] else "secondary"
    ), gr.Button(variant="secondary")


def draw_example(examples: list[str]) -> str:
    random.shuffle(examples)
    return examples[0]


async def check(statement: str, last_result: EvidenceSeekerResult):
    request_time = datetime.now(timezone.utc)
    last_result.request_time = request_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    last_result.request = statement

    if UI_TEST_MODE:
        last_result.claims = [claim.model_dump() for claim in _dummy_claims]
    else:
        logger.log("INFO", f"Checking '{statement}'... This could take a while.")
        checked_claims = await EVIDENCE_SEEKER(statement)
        last_result.claims = checked_claims

    result = result_as_markdown(
        ev_result=last_result,
        translations=APP_CONFIG.translations[APP_CONFIG.language],
        jinja2_md_template=APP_CONFIG.md_template
    )

    logger.info(
        f"Result of statement '{statement}' checked (uid: {last_result.request_uid})",
    )
    return result, last_result


EVIDENCE_SEEKER = EvidenceSeeker(
    preprocessing_config_file=APP_CONFIG.preprocessing_config_file,
    retrieval_config_file=APP_CONFIG.retrieval_config_file,
    confirmation_analysis_config_file=APP_CONFIG.confirmation_analysis_config_file,
)

with gr.Blocks(title="EvidenceSeeker") as demo:
    last_result = gr.State(
        EvidenceSeekerResult(
            retrieval_config=EVIDENCE_SEEKER.retriever.config,
            confirmation_config=EVIDENCE_SEEKER.analyzer.config,
            preprocessing_config=EVIDENCE_SEEKER.preprocessor.config,
        )
    )
    examples = gr.State(APP_CONFIG.examples)
    password_authenticated = gr.State(
        value=False if APP_CONFIG.password_protection else True
    )
    if APP_CONFIG.force_agreement:
        allow_result_persistance = gr.State(value=False)
        read_warning = gr.State(value=False)
    else:
        allow_result_persistance = gr.State(value=True)
        read_warning = gr.State(value=True)

    good = gr.Button(value="👍", visible=False, interactive=False, render=False)
    bad = gr.Button(value="👎", visible=False, interactive=False, render=False)
    feedback = gr.Markdown(
        value="Wie zufrieden bist du mit der Antwort?", render=False, visible=False
    )

    @gr.render(inputs=[password_authenticated, read_warning, allow_result_persistance])
    def renderApp(
        password_authenticated_val: bool,
        read_warning_val: bool,
        allow_result_persistance_val: bool
    ):
        if password_authenticated_val and read_warning_val:
            gr.Markdown(
                """
                    # 🕵️‍♀️ EvidenceSeeker DemoApp
                    Gib eine Aussage in das Textfeld ein und lass sie durch den EvidenceSeeker prüfen:
                """,
                key="intro",
            )
            with gr.Row():
                statement = gr.Textbox(
                    value="",
                    label="Zu prüfende Aussage:",
                    interactive=True,
                    scale=10,
                    lines=3,
                    # key="statement",
                )
                with gr.Column(scale=1):
                    example_btn = gr.Button(
                        "Zufälliges Beispiel", 
                        # key="example_btn"
                    )
                    check_btn = gr.Button(
                        "Prüfe Aussage",
                        interactive=False,
                        # key="submit_btn"
                    )
            result = gr.Markdown(
                "", 
                min_height=80,
                # key="results",
                show_copy_button=True
            )
            with gr.Column():
                feedback.render()
                with gr.Row():
                    good.render()
                    bad.render()

            def logging(evse_result):
                if allow_result_persistance_val:
                    log_result(
                        evse_result=evse_result,
                        result_dir=APP_CONFIG.result_dir,
                        local_base=APP_CONFIG.local_base,
                        subdirectory_construction=APP_CONFIG.subdirectory_construction,
                        write_on_github=APP_CONFIG.write_on_github,
                        github_token_name=APP_CONFIG.github_token_name,
                        repo_name=APP_CONFIG.repo_name,
                    )

            check_btn.click(deactivate, None, [check_btn, good, bad, feedback]).then(
                (
                    lambda: "### Aussage wird geprüft... Dies könnte ein paar Minuten dauern."
                ),
                None,
                result,
            ).then(check, [statement, last_result], [result, last_result]).then(
                logging, last_result, None
            ).then(
                reactivate, [check_btn, statement], [feedback, check_btn, good, bad]
            )
            good.click(log_feedback, [good, last_result], [good, bad]).then(
                logging, [last_result], None
            )
            bad.click(log_feedback, [bad, last_result], [bad, good]).then(
                logging, [last_result], None
            )
            example_btn.click(fn=draw_example, inputs=examples, outputs=statement)
            statement.change(
                lambda s: (
                    gr.Button("Prüfe Aussage", interactive=False) #, key="submit_btn")
                    if s.strip() == ""
                    else gr.Button("Prüfe Aussage", interactive=True) #,key="submit_btn")
                ),
                statement,
                [check_btn],
            )
        elif password_authenticated_val:
            gr.Markdown("# Datenschutzhinweis & Disclaimer")
            gr.HTML(
                f"""
                <div style="background-color:#fff7ed;padding:25px;border-radius: 10px;">
                     <p>⚠️ <b>Achtung</b></p>
                    <p>{APP_CONFIG.warning_text}</p>
                </div>
                """
            )
            gr.Markdown(APP_CONFIG.consent_text)
            consent_box = gr.Checkbox(
                False,
                label=APP_CONFIG.consent_text,
                info="**Einwilligung zur Datenweiterverarbeitung (Optional)**",
            )
            agree_button = gr.Button("Ich habe die Hinweise zur Kenntnis genommen")
            agree_button.click(
                lambda consent_save_res: (True, consent_save_res),
                inputs=consent_box,
                outputs=[read_warning, allow_result_persistance]
            )
        else:
            gr.Markdown(
                """
                    # 🕵️ EvidenceSeeker DemoApp
                """
            )
            box = gr.Textbox(
                label="Bitte für Zugriff Passwort eingeben",
                autofocus=True,
                type="password",
                submit_btn=True,
            )
            res = gr.Markdown(value="")
            box.submit(
                auth,
                inputs=[box, password_authenticated],
                outputs=[res, password_authenticated, box]
            )


demo.launch()
