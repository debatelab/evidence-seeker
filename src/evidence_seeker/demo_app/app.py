from __future__ import annotations

import gradio as gr
import random
import os
from loguru import logger
from datetime import datetime, timezone
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
import uuid

from evidence_seeker.demo_app.app_config import AppConfig

from evidence_seeker import (
    EvidenceSeeker,
    EvidenceSeekerResult,
    result_as_markdown,
    log_result,
    ConfirmationLevel,
    _DUMMY_CLAIMS
)

from evidence_seeker import (
    CheckedClaim,
    Document,
    StatementType
)

# ### APP CONFIGURATION ####
# dotenv.load_dotenv()

config_file_path = os.getenv("APP_CONFIG_FILE", None)

if config_file_path is None:
    raise ValueError("Missing configruation file.")
APP_CONFIG = AppConfig.from_file(config_file_path)
ui = APP_CONFIG.ui_texts.get_texts(APP_CONFIG.language)

UI_TEST_MODE = os.getenv("UI_TEST_MODE", False)

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
        output = ui.server_error
    elif not check_password(pw, os.environ[APP_CONFIG.password_env_name]):
        output = ui.wrong_password
    else:
        output = ui.continue_text
        password_authenticated = True
    return output, password_authenticated, b


def reactivate(check_btn, statement, checked):
    if statement.strip() != "":
        check_btn = gr.Button(visible=True, interactive=True)
    good = gr.Button(visible=checked, interactive=checked)
    bad = gr.Button(visible=checked, interactive=checked)
    feedback = gr.Markdown(visible=checked)
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
    if examples:
        random.shuffle(examples)
        return examples[0]
    else:
        return ""

async def check(statement: str, last_result: EvidenceSeekerResult):
    global EVIDENCE_SEEKER
    request_time = datetime.now(timezone.utc)
    checked = False
    last_result = EvidenceSeekerResult(request = statement,
                                       time=request_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                       preprocessing_config=EVIDENCE_SEEKER.preprocessor.config,
                                       retrieval_config=EVIDENCE_SEEKER.retriever.config,
                                       confirmation_config=EVIDENCE_SEEKER.analyzer.config)
    if UI_TEST_MODE:
        last_result.claims = _DUMMY_CLAIMS
        result = result_as_markdown(
                evse_result=last_result,
                translations=APP_CONFIG.translations[APP_CONFIG.language],
                jinja2_md_template=APP_CONFIG.md_template,
            )
        checked = True
    else:
        logger.log("INFO", f"Checking '{statement}'... This could take a while.")
        try:
            checked_claims = await EVIDENCE_SEEKER(statement)
            last_result.claims = checked_claims
            result = result_as_markdown(
                evse_result=last_result,
                translations=APP_CONFIG.translations[APP_CONFIG.language],
                jinja2_md_template=APP_CONFIG.md_template,
            )

            logger.info(
                f"Result of statement '{statement}' checked (uid: {last_result.uid})",
            )
            checked = True
        except Exception as e:
            result = APP_CONFIG.get_ui_texts().server_error
            logger.error(
                f"Exception occured while checking '{statement}' ({e})",
            )
    return result, last_result, checked

logger.info(
    "Using the following config files to initiate pipeline:"
    f"{APP_CONFIG.retrieval_config_file}\n"
    f"{APP_CONFIG.confirmation_analysis_config_file}\n"
    f"{APP_CONFIG.preprocessing_config_file}\n"
)
EVIDENCE_SEEKER = EvidenceSeeker(
    preprocessing_config_file=APP_CONFIG.preprocessing_config_file,
    retrieval_config_file=APP_CONFIG.retrieval_config_file,
    confirmation_analysis_config_file=APP_CONFIG.confirmation_analysis_config_file,
)

with gr.Blocks(title="EvidenceSeeker") as evse_demo_app:
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
    checked = gr.State(value=False)
    if APP_CONFIG.force_agreement:
        allow_result_persistance = gr.State(value=False)
        read_warning = gr.State(value=False)
    else:
        allow_result_persistance = gr.State(value=True)
        read_warning = gr.State(value=True)

    good = gr.Button(value="👍", visible=False, interactive=False, render=False)
    bad = gr.Button(value="👎", visible=False, interactive=False, render=False)
    feedback = gr.Markdown(
        value=ui.feedback_question, render=False, visible=False
    )

    @gr.render(inputs=[password_authenticated, read_warning, allow_result_persistance])
    def renderApp(
        password_authenticated_val: bool,
        read_warning_val: bool,
        allow_result_persistance_val: bool
    ):
        if password_authenticated_val and read_warning_val:
            gr.Markdown(
                f"# {ui.title}\n"
                # f"{ui['info']}\n\n"
                # f"{ui['description']}"
            )
            gr.HTML(ui.info)
            gr.Markdown(ui.description)
            with gr.Row():
                statement = gr.Textbox(
                    value="",
                    label=ui.statement_label,
                    interactive=True,
                    scale=10,
                    lines=3,
                )
                with gr.Column(scale=1):

                    example_btn = gr.Button(ui.random_example)
                    check_btn = gr.Button(
                        ui.check_statement,
                        interactive=False,
                    )
            result = gr.Markdown(
                "",
                min_height=80,
                show_copy_button=True
            )
            with gr.Column():
                feedback.render()
                with gr.Row():
                    good.render()
                    bad.render()

            def logging(evse_result, checked):
                if allow_result_persistance_val and checked:
                    log_result(
                        evse_result=evse_result,
                        result_dir=APP_CONFIG.result_dir,
                        local_base=APP_CONFIG.local_base if APP_CONFIG.local_base else "",
                        subdirectory_construction=APP_CONFIG.subdirectory_construction,
                        write_on_github=APP_CONFIG.write_on_github,
                        github_token_name=APP_CONFIG.github_token_name,
                        repo_name=APP_CONFIG.repo_name,
                        additional_markdown_log=APP_CONFIG.save_markdown,
                    )

            check_btn.click(deactivate, None, [check_btn, good, bad, feedback]).then(
                (
                    lambda: ui.checking_message
                ),
                None,
                result,
            ).then(check, [statement, last_result], [result, last_result, checked]).then(
                logging, [last_result, checked], None
            ).then(
                reactivate, [check_btn, statement, checked], [feedback, check_btn, good, bad]
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
                    gr.Button(ui.check_statement, interactive=False)
                    if s.strip() == ""
                    else gr.Button(ui.check_statement, interactive=True)
                ),
                statement,
                [check_btn],
            )
        elif password_authenticated_val:
            gr.Markdown(f"# {ui.privacy_title}")
            gr.HTML(ui.disclaimer_text)
            gr.HTML(ui.data_policy_text)
            consent_box = gr.Checkbox(
                False,
                label=ui.consent_text,
                info=ui.consent_info,
            )
            agree_button = gr.Button(ui.agree_button)
            agree_button.click(
                lambda consent_save_res: (True, consent_save_res),
                inputs=consent_box,
                outputs=[read_warning, allow_result_persistance]
            )
        else:
            gr.Markdown(f"# {ui.title}")
            box = gr.Textbox(
                label=ui.password_label,
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


evse_demo_app.launch()
