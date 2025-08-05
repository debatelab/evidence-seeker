from __future__ import annotations

import gradio as gr
import random
import dotenv
import os
from loguru import logger
from typing import Any
from datetime import datetime, timezone
import yaml
import uuid
from glob import glob
from jinja2 import Environment, FileSystemLoader, Template
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from github import Github, Auth, UnknownObjectException


from evidence_seeker.demo_app.app_config import AppConfig

from evidence_seeker import EvidenceSeeker, EvidenceSeekerResult


from evidence_seeker import (
    CheckedClaim,
    Document,
    StatementType
)

dotenv.load_dotenv()

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

def setup_from_app_config() -> EvidenceSeeker | None:
    global APP_CONFIG
    configs = {
        "preprocessing_config_file": APP_CONFIG.preprocessing_config_file,
        "retrieval_config_file": APP_CONFIG.retrieval_config_file,
        "confirmation_analysis_config_file": APP_CONFIG.confirmation_analysis_config_file,
    }
    if APP_CONFIG.dummy:
        logger.info("Running in dummy mode, no EvidenceSeeker instance created.")
        return None
    else:
        return EvidenceSeeker(**configs)


def get_sources(documents, confirmation_by_document) -> str | None:
    grouped = {}
    for doc in documents:
        if doc["metadata"]["file_name"] not in grouped.keys():
            grouped[doc["metadata"]["file_name"]] = {
                "author": doc["metadata"]["author"],
                "url": doc["metadata"]["url"],
                "title": doc["metadata"]["title"].replace("{", "").replace("}", ""),
                "texts": [
                    {
                        "original_text": doc["metadata"]["original_text"],
                        "conf": confirmation_by_document[doc["uid"]],
                        "full_text": doc["text"],
                    }
                ],
            }
        else:
            grouped[doc["metadata"]["file_name"]]["texts"].append(
                {
                    "original_text": doc["metadata"]["original_text"],
                    "conf": confirmation_by_document[doc["uid"]],
                    "full_text": doc["text"],
                }
            )

    t = []
    for doc in grouped.keys():
        grouped[doc]["texts"] = sorted(
            grouped[doc]["texts"], key=lambda item: item["conf"], reverse=True
        )
        t.append(
            f"- {grouped[doc]['author']}: *{grouped[doc]['title']}* ([Link]({grouped[doc]['url']})):"
        )
        for text in grouped[doc]["texts"]:
            orig = text["original_text"].strip().replace("\n", " ").replace('"', "'")
            short = f'"{orig}" **[{round(text["conf"],5)}]**'
            detailed = (
                '"  '
                + text["full_text"].strip().replace("\n", "  ").replace('"', "'")
                + '"'
            )
            part = f"  - {short}\n    <details>\n    <summary>Mehr Details</summary>\n\n    - {detailed}\n\n    </details>"
            t.append(part)
    if len(t) == 0:
        return None
    else:
        t = "\n\n".join(t) + "\n\n"
        return (
            "\n\n<details>\n\n<summary>Quellenverweise</summary>\n\n"
            + t
            + "</details>\n\n"
        )


def describe(ev_result: EvidenceSeekerResult) -> str:
    global APP_CONFIG
    claims = [
        (claim, get_sources(claim["documents"], claim["confirmation_by_document"]))
        for claim in ev_result.claims
    ]
    translation = APP_CONFIG.translation
    result_template = Template(APP_CONFIG.md_template)
    md = result_template.render(
        feedback=ev_result.feedback["binary"],
        statement=ev_result.request,
        time=ev_result.request_time,
        claims=claims,
        translation=translation,
    )
    return md


def check_password(input_password: str, hash: str) -> bool:
    ph = PasswordHasher()
    try:
        ph.verify(hash, input_password)
        return True
    except VerificationError:
        return False


def auth(pw: str, valid: bool):
    output = ""
    b = gr.Textbox(value="")
    if "APP_HASH" not in os.environ:
        output = "Etwas ist auf unserer Seite schiefgegangen :-("
    elif not check_password(pw, os.environ["APP_HASH"]):
        output = "Falsches Passwort. Bitte versuche es erneut."
    else:
        output = "Weiter..."
        valid = True
    return output, valid, b


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
    global EVIDENCE_SEEKER
    request_time = datetime.now(timezone.utc)
    last_result.request_time = request_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    last_result.request = statement
    last_result.request_uid = str(uuid.uuid4())
    last_result.feedback["binary"] = None
    if EVIDENCE_SEEKER is not None:
        logger.log("INFO", f"Checking '{statement}'... This could take a while.")
        checked_claims = await EVIDENCE_SEEKER(statement)
        last_result.claims = checked_claims
        result = describe(last_result)  # describe_result(statement, checked_claims)
    else:
        last_result.claims = [claim.model_dump() for claim in _dummy_claims]
        # result = gr.Markdown(value=f"{last_result.model_dump()}")
        result = describe(last_result)
    logger.log(
        "INFO",
        f"Result of statement '{statement}' checked (request {last_result.request_uid})",
    )
    return result, last_result


def initialize_result_state():
    global APP_CONFIG, EVIDENCE_SEEKER
    conf: dict[str, Any] = APP_CONFIG.model_dump()
    if EVIDENCE_SEEKER is not None:
        conf["retrieval_config"] = EVIDENCE_SEEKER.retriever.config
        conf["confirmation_config"] = EVIDENCE_SEEKER.analyzer.workflow.config
        conf["preprocessing_config"] = EVIDENCE_SEEKER.preprocessor.workflow.config
    else:
        conf["retrieval_conf"] = None
        conf["confirmation_config"] = None
        conf["preprocessing_config"] = None
    last_result = EvidenceSeekerResult(**conf)
    return gr.State(last_result)


def _current_subdir(subdirectory_construction: str | None) -> str:
    if subdirectory_construction is None:
        return ""
    now = datetime.now()
    if subdirectory_construction == "monthly":
        subdirectory_path = now.strftime("y%Y_m%m")
    elif subdirectory_construction == "weekly":
        year, week, _ = now.isocalendar()
        subdirectory_path = f"y{year}_w{week}"
    elif subdirectory_construction == "yearly":
        subdirectory_path = now.strftime("y%Y")
    elif subdirectory_construction == "daily":
        subdirectory_path = now.strftime("%Y_%m_%d")
    else:
        subdirectory_path = ""
    return subdirectory_path


def log_result(last_result: EvidenceSeekerResult):
    global RUNS_ON_SPACES, APP_CONFIG

    # Do not log results if pipeline failed somehow
    if len(last_result.claims) == 0:
        return

    assert last_result.request_time is not None
    ts = datetime.strptime(last_result.request_time, "%Y-%m-%d %H:%M:%S UTC").strftime(
        "%Y_%m_%d"
    )
    fn = f"request_{ts}_{last_result.request_uid}.yaml"
    subdir = _current_subdir(APP_CONFIG.subdirectory_construction)
    if RUNS_ON_SPACES:
        result_dir = APP_CONFIG.result_dir
        filepath = (
            "/".join([result_dir, fn])
            if subdir == ""
            else "/".join([result_dir, subdir, fn])
        )
        assert (
            APP_CONFIG.github_token is not None
            and APP_CONFIG.github_token in os.environ.keys()
        )
        auth = Auth.Token(os.environ[APP_CONFIG.github_token])
        g = Github(auth=auth)
        repo = g.get_repo(APP_CONFIG.repo_name)
        content = last_result.yaml_dump(stream=None)
        try:
            c = repo.get_contents(filepath)
            repo.update_file(
                filepath, f"Update result ({last_result.request_uid})", content, c.sha
            )
        except UnknownObjectException:
            repo.create_file(
                filepath, f"Upload new result ({last_result.request_uid})", content
            )
        return
    else:
        local_base = APP_CONFIG.local_base if APP_CONFIG.local_base else ""
        result_dir = APP_CONFIG.result_dir if APP_CONFIG.result_dir else ""
        files = glob("/".join([local_base, result_dir, "**", fn]), recursive=True)
        assert len(files) < 2
        if len(files) == 0:
            filepath = "/".join([local_base, result_dir, subdir, fn])
            os.makedirs("/".join([local_base, result_dir, subdir]), exist_ok=True)
        else:
            filepath = files[0]
        with open(filepath, encoding="utf-8", mode="w") as f:
            last_result.yaml_dump(f)


if "RUNS_ON_SPACES" not in os.environ.keys():
    RUNS_ON_SPACES = False
    logger.info("Gradioapp runs locally.")
else:
    RUNS_ON_SPACES = os.environ["RUNS_ON_SPACES"] == "True"
    logger.info("Gradioapp runs on 🤗.")

if "APP_CONFIG_FILE" in os.environ.keys():
    with open(os.environ["APP_CONFIG_FILE"]) as f:
        APP_CONFIG = AppConfig(**yaml.safe_load(f))
else:
    APP_CONFIG = AppConfig()

EVIDENCE_SEEKER = setup_from_app_config()

with gr.Blocks(title="EvidenceSeeker") as demo:
    last_result = initialize_result_state()
    examples = gr.State(APP_CONFIG.examples)
    valid = gr.State(value=not RUNS_ON_SPACES)
    agreements = gr.State(value=(not RUNS_ON_SPACES, not RUNS_ON_SPACES))

    good = gr.Button(value="👍", visible=False, interactive=False, render=False)
    bad = gr.Button(value="👎", visible=False, interactive=False, render=False)
    feedback = gr.Markdown(
        value="Wie zufrieden bist du mit der Antwort?", render=False, visible=False
    )

    @gr.render(inputs=[valid, agreements])
    def renderApp(v: bool, a: tuple[bool, bool]):
        if v and a[0]:
            gr.Markdown(
                """
                    # 🕵️‍♀️ EvidenceSeeker DemoApp
                    Gib eine Aussage in das Textfeld ein und lass sie durch den EvidenceSeeker prüfen:
                """,
                key="intro",
            )
            with gr.Row():
                statement = gr.Textbox(
                    value=f"",
                    label="Zu prüfende Aussage:",
                    interactive=True,
                    scale=10,
                    lines=3,
                    key="statement",
                )
                with gr.Column(scale=1):
                    example_btn = gr.Button("Zufälliges Beispiel", key="example_btn")
                    check_btn = gr.Button(
                        "Prüfe Aussage", interactive=False, key="submit_btn"
                    )
            result = gr.Markdown(
                "", min_height=80, key="results", show_copy_button=True
            )
            with gr.Column():
                feedback.render()
                with gr.Row():
                    good.render()
                    bad.render()

            def logging(r):
                if a[1]:
                    log_result(r)

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
                    gr.Button("Prüfe Aussage", interactive=False, key="submit_btn")
                    if s.strip() == ""
                    else gr.Button("Prüfe Aussage", interactive=True, key="submit_btn")
                ),
                statement,
                [check_btn],
            )
        elif v:
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
            agree_button.click(lambda c: (True, c), consent_box, agreements)
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
            box.submit(auth, inputs=[box, valid], outputs=[res, valid, box])


demo.launch()
