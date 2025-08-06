"utils.py"

from typing import Dict, List
from jinja2 import Environment
from jinja2 import Template
from typing import Any
from datetime import datetime
import os
from glob import glob
from github import Github, Auth, UnknownObjectException
from enum import Enum

from .results import EvidenceSeekerResult


class ConfirmationLevel(Enum):
    STRONGLY_CONFIRMED = "strongly_confirmed"
    CONFIRMED = "confirmed"
    WEAKLY_CONFIRMED = "weakly_confirmed"
    INCONCLUSIVE_CONFIRMATION = "inconclusive_confirmation"
    WEAKLY_DISCONFIRMED = "weakly_disconfirmed"
    DISCONFIRMED = "disconfirmed"
    STRONGLY_DISCONFIRMED = "strongly_disconfirmed"


def confirmation_level(degree_of_confirmation: float) -> ConfirmationLevel:
    if degree_of_confirmation > 0.6:
        return ConfirmationLevel.STRONGLY_CONFIRMED
    if degree_of_confirmation > 0.4:
        return ConfirmationLevel.CONFIRMED
    if degree_of_confirmation > 0.2:
        return ConfirmationLevel.WEAKLY_CONFIRMED
    if degree_of_confirmation < -0.6:
        return ConfirmationLevel.STRONGLY_DISCONFIRMED
    if degree_of_confirmation < -0.4:
        return ConfirmationLevel.DISCONFIRMED
    if degree_of_confirmation < -0.2:
        return ConfirmationLevel.WEAKLY_DISCONFIRMED
    return ConfirmationLevel.INCONCLUSIVE_CONFIRMATION

# TODO (refactor!): Using this function is very ideosynctratic
# since it hinges on specfici meta-data (which is
# specific to the EvSe Demo Dataset).
def get_grouped_sources(
        documents,
        confirmation_by_document
) -> Dict[str, Dict[str, Any]]:
    docs_grouped_by_src_file = {}
    # Group documents by filenname
    for doc in documents:
        file_name = doc["metadata"].get("file_name", None)
        if file_name is None:
            raise ValueError("No filename found in metadata")
        else:
            if file_name not in docs_grouped_by_src_file:
                docs_grouped_by_src_file[file_name] = {
                    "author": doc["metadata"]["author"],
                    "url": doc["metadata"]["url"],
                    "title": doc["metadata"]["title"].replace("{", "").replace("}", ""),
                    "texts": [],
                }
            docs_grouped_by_src_file[file_name]["texts"].append({
                "original_text": (
                    doc["metadata"]["original_text"]
                    .strip()
                    .replace("\n", " ")
                    .replace('"', "'")
                ),
                "conf": confirmation_by_document[doc["uid"]],
                "conf_level": confirmation_level(confirmation_by_document[doc["uid"]]).value,
                "full_text": (
                    doc["text"]
                    .strip()
                    .replace("\n", "  ")
                    .replace('"', "'")
                ),
            })
    # Sort texts by confidence score (highest first)
    for file_name in docs_grouped_by_src_file:
        docs_grouped_by_src_file[file_name]["texts"] = sorted(
            docs_grouped_by_src_file[file_name]["texts"],
            key=lambda item: item["conf"],
            reverse=True
        )
    return docs_grouped_by_src_file


def result_as_markdown(
    ev_result: EvidenceSeekerResult,
    translations: dict[str, str],
    jinja2_md_template: str
) -> str:
    # TODO: see task from `get_grouped_sources`
    claims = [
        (
            claim,
            get_grouped_sources(
                claim["documents"],
                claim["confirmation_by_document"]
            )
        )
        for claim in ev_result.claims
    ]
    result_template = Template(jinja2_md_template)
    md = result_template.render(
        feedback=ev_result.feedback["binary"],
        statement=ev_result.request,
        time=ev_result.request_time,
        claims=claims,
        translation=translations,
    )
    return md


# TODO: Use enum type
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


def log_result(
    evse_result: EvidenceSeekerResult,
    result_dir: str,
    local_base: str = ".",
    subdirectory_construction: str | None = None, 
    write_on_github: bool = False,
    github_token_name: str | None = None,
    repo_name: str | None = None,
):
    # Do not log results if pipeline failed somehow
    # TODO: Better to use state field (in 'EvSeResult') by catching
    # errors and/or accessing the error codes from request
    # (refactor workflows or pipeline for this)
    if len(evse_result.claims) == 0:
        return
    if evse_result.request_time is None:
        raise ValueError("Request time not set in result.")
    # constructing file name
    ts = datetime.strptime(
        evse_result.request_time, "%Y-%m-%d %H:%M:%S UTC"
    ).strftime("%Y_%m_%d")
    fn = f"request_{ts}_{evse_result.request_uid}.yaml"
    subdir = _current_subdir(subdirectory_construction)
    if write_on_github:
        result_dir = result_dir
        filepath = (
            "/".join([result_dir, fn])
            if subdir == ""
            else "/".join([result_dir, subdir, fn])
        )
        if (
            github_token_name is None
            or github_token_name not in os.environ.keys()
        ):
            raise ValueError(
                "Github token name not set or token not"
                "found as env variable by the specified name."
            )
        auth = Auth.Token(os.environ[github_token_name])
        g = Github(auth=auth)
        repo = g.get_repo(repo_name)
        content = evse_result.yaml_dump(stream=None)
        try:
            c = repo.get_contents(filepath)
            repo.update_file(
                filepath,
                f"Update result ({evse_result.request_uid})",
                content,
                c.sha
            )
        except UnknownObjectException:
            repo.create_file(
                filepath,
                f"Upload new result ({evse_result.request_uid})",
                content
            )
        return
    else:
        # TODO: check whether this works with relative paths configs
        if result_dir is None:
            result_dir = ""
        files = glob("/".join([local_base, result_dir, "**", fn]), recursive=True)
        assert len(files) < 2
        if len(files) == 0:
            filepath = "/".join([local_base, result_dir, subdir, fn])
            os.makedirs("/".join([local_base, result_dir, subdir]), exist_ok=True)
        else:
            filepath = files[0]
        with open(filepath, encoding="utf-8", mode="w") as f:
            evse_result.yaml_dump(f)


_md_template_str = """
# EvidenceSeeker Results

*Number of claims submitted:* {{ input_results_tuples |length }}
{% for input, output in input_results_tuples %}
## Input: {{ input }}
**Submitted claim:** {{ input }}
### Results
{% for clarified_claim in output %}

#### "{{ clarified_claim['text'] }}"

**Clarified claim:** <font color="orange">{{ clarified_claim['text'] }}</font> [type: {{clarified_claim['statement_type'].value}}]

**Status**: {{clarified_claim['verbalized_confirmation']}}

|Metric|Value|
|:---|---:|
|Average confirmation|{{ clarified_claim['average_confirmation'] | round(2) }}|
|Evidential divergence|{{clarified_claim['evidential_uncertainty'] | round(2) }}|
|Width of evidential base|{{clarified_claim['n_evidence']}}|


{% if show_documents %}
**Documents:**
{% for document in clarified_claim['documents'] %}

+ {{ document['text'] }}
  - **Confirmation**: {{ clarified_claim['confirmation_by_document'][document['uid']] | round(3) }}

{% endfor %}
{% endif %}

{% endfor %}

{% endfor %}
"""


def results_to_markdown(
        input_list: List[str],
        results_list: List[List[Dict]],
        show_documents: bool = False) -> str:
    env = Environment()
    md_template = env.from_string(_md_template_str)
    markdown = md_template.render(
        input_results_tuples=list(zip(input_list, results_list)),
        show_documents=show_documents
    )
    return markdown

def describe_result(input, results) -> str:
    preamble_template = (
        '## EvidenceSeeker Results\n\n'
        '### Input\n\n'
        '**Submitted claim:** {claim}\n\n'
        '### Results\n\n'
    )
    result_template = (
        '**Clarified claim:** <font color="orange">{text}</font> [_{statement_type}_]\n\n'
        '**Status**: {verbalized_confirmation}\n\n'
        '|Metric|Value|\n'
        '|:---|---:|\n'
        '|Average confirmation|{average_confirmation:.3f}|\n'
        '|Evidential divergence|{evidential_uncertainty:.3f}|\n'
        '|Width of evidential base|{n_evidence}|\n\n'
    )
    markdown = []
    markdown.append(preamble_template.format(claim=input))
    for claim_dict in results:
        rdict = claim_dict.copy()
        rdict["statement_type"] = rdict["statement_type"].value
        markdown.append(result_template.format(**claim_dict))
    return "\n".join(markdown)

