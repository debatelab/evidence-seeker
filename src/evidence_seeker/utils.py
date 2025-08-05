"utils.py"

from typing import Dict, List
from jinja2 import Environment
from jinja2 import Template
from typing import Any

from .results import EvidenceSeekerResult


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

