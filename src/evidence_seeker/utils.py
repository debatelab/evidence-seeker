"utils.py"

import enum
from typing import Callable
from jinja2 import Template
from typing import Any, Mapping, Dict
from datetime import datetime
import os
from glob import glob
from github import Github, Auth, UnknownObjectException
import importlib.resources as pkg_resources
from loguru import logger

from .results import EvidenceSeekerResult
from .confirmation_aggregation.base import (
    confirmation_level
)
from .datamodels import Document, StatementType, CheckedClaim, ConfirmationLevel

_PACKAGE_DATA_MODULE = "evidence_seeker.package_data"
_DEFAULT_MD_TEMPLATE = "templates/default_markdown.tmpl"

def result_as_markdown(
    evse_result: EvidenceSeekerResult,
    translations: dict[str, str] = dict(),
    jinja2_md_template: str | Template | None = None,
    show_documents: bool = True,
    **kwargs,
) -> str:
    # use simple template from package if
    # none is given
    if jinja2_md_template is None:
        template_path = pkg_resources.files(
            _PACKAGE_DATA_MODULE
        ).joinpath(_DEFAULT_MD_TEMPLATE)

        if not os.path.exists(str(template_path)):
            raise ValueError(
                "Template file not found or unreadable in package data module."
            )
        with open(str(template_path), encoding="utf-8") as f:
            jinja2_md_template = f.read()
            result_template = Template(jinja2_md_template)
    elif isinstance(jinja2_md_template, str):
        result_template = Template(jinja2_md_template)
    elif isinstance(jinja2_md_template, Template):
        result_template = jinja2_md_template
    else:
        raise ValueError(
            "The template must be of type 'template', 'str' or 'None'."
        )

    md = result_template.render(
        evse_result=evse_result,
        feedback=evse_result.feedback["binary"],
        statement=evse_result.request,
        time=evse_result.time,
        claims=evse_result.claims,
        translation=translations,
        show_documents=show_documents,
        confirmation_level=confirmation_level,
        **kwargs,
    )
    return md

class SubdirConstruction(enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

    @classmethod
    def value2formatcode(cls) -> Dict[str, str]:
        return {
            cls.DAILY.value : "%Y_%m_%d",
            cls.WEEKLY.value : "y%Y_w%W",
            cls.MONTHLY.value : "y%Y_m%m",
            cls.YEARLY.value : "y%Y",
        }

def _current_subdir(subdirectory_construction: str | None) -> str:
    if subdirectory_construction is None or subdirectory_construction not in SubdirConstruction.value2formatcode().keys():
        return ""
    now = datetime.now()
    dateformat = SubdirConstruction.value2formatcode()[subdirectory_construction]
    return now.strftime(dateformat)

# TODO: provision of md template via argument
def log_result(
    evse_result: EvidenceSeekerResult,
    result_dir: str = "",
    local_base: str = ".",
    subdirectory_construction: str | None = None,
    write_on_github: bool = False,
    github_token_name: str | None = None,
    repo_name: str | None = None,
    additional_markdown_log: bool = False,
    filename_without_suffix: Callable[[EvidenceSeekerResult], str] | str | None = None,
):
    if evse_result.time is None:
        raise ValueError("Request time not set in result.")
    # constructing file name
    if filename_without_suffix is None:
        ts = datetime.strptime(
            evse_result.time, "%Y-%m-%d %H:%M:%S UTC"
        ).strftime("%Y_%m_%d")
        filename_without_suffix = f"{ts}_{evse_result.uid}"
    elif callable(filename_without_suffix):
        filename_without_suffix = filename_without_suffix(evse_result)

    fn = f"{filename_without_suffix}.yaml"
    md_fn = f"{filename_without_suffix}.md"

    subdir = _current_subdir(subdirectory_construction)
    if write_on_github and repo_name:
        filepath = os.path.join(result_dir, subdir, fn)
        md_filepath = os.path.join(result_dir, subdir, md_fn)
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
        logger.info(
            "Log evidence seeker result to "
            f"{filepath} "
            f"(with additional md log)" if additional_markdown_log else ""
            f"in repo '{repo}' on github."
        )
        content = evse_result.yaml_dump(stream=None)
        md_content = result_as_markdown(evse_result)
        try:
            c = repo.get_contents(filepath)
            repo.update_file(
                filepath,
                f"Update result ({evse_result.uid})",
                content,
                c.sha
            )
            if additional_markdown_log:
                c = repo.get_contents(md_filepath)
                repo.update_file(
                    md_filepath,
                    f"Update result ({evse_result.uid})",
                    md_content,
                    c.sha
                )
        except UnknownObjectException:
            repo.create_file(
                filepath,
                f"Upload new result ({evse_result.uid})",
                content
            )
            if additional_markdown_log:
                repo.create_file(
                    md_filepath,
                    f"Upload new result ({evse_result.uid})",
                    md_content
                )
        return
    else:
        files = glob(os.path.join(local_base, result_dir, "**", fn),
                     recursive=True)
        if len(files) < 2:
            if len(files) == 0:
                filepath = os.path.join(local_base, result_dir, subdir, fn)
                md_filepath = os.path.join(
                    local_base, result_dir, subdir, md_fn
                )
                os.makedirs(os.path.join(local_base, result_dir, subdir), 
                            exist_ok=True)
            else:
                filepath = files[0]
                md_filepath = filepath.replace(".yaml", ".md")
            logger.info(
                "Log evidence seeker result to "
                f"{filepath} "
                f"(with additional md log)" if additional_markdown_log else ""
            )
            with open(filepath, encoding="utf-8", mode="w") as f:
                evse_result.yaml_dump(f)
            if additional_markdown_log:
                with open(md_filepath, encoding="utf-8", mode="w") as f:
                    f.write(result_as_markdown(evse_result))
        else:
            raise Exception("The uid of the result is not unique.")

_DUMMY_DOCS = [
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

_DUMMY_CLAIMS = [
    CheckedClaim(
        text="The AMOC is slowing down",
        negation="The AMOC is not changing",
        uid="123",
        documents=_DUMMY_DOCS,
        n_evidence=2,
        statement_type=StatementType.DESCRIPTIVE,
        average_confirmation=0.2,
        confirmation_level=ConfirmationLevel.WEAKLY_CONFIRMED,
        evidential_uncertainty=0.1,
        verbalized_confirmation="The claim is weakly confirmed.",
        confirmation_by_document={
            "1f47ce98-4105-4ddc-98a9-c4956dab2000": 0.1,
            "6fcd6c0f-99a1-48e7-881f-f79758c54769": 0.3,
        },
    ),
]