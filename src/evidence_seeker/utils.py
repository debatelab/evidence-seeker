"utils.py"

from typing import Callable
from jinja2 import Template
from typing import Any, Mapping
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
from .datamodels import Document

_PACKAGE_DATA_MODULE = "evidence_seeker.package_data"
_DEFAULT_MD_TEMPLATE = "templates/default_markdown.tmpl"

# TODO (refactor!): Using this function is very ideosynctratic
# since it hinges on specfici meta-data (which is
# specific to the EvSe Demo Dataset).


def get_grouped_sources(
        documents: list[Document] | None,
        confirmation_by_document: Mapping[str, float] | None
) -> dict[str, dict[str, Any]]:
    if documents is None or confirmation_by_document is None:
        return dict()
    docs_grouped_by_src_file = {}
    # Group documents by filenname
    for doc in documents:
        file_name = doc.metadata.get("file_name", None)
        if file_name is None:
            raise ValueError("No filename found in metadata")
        else:
            if file_name not in docs_grouped_by_src_file:
                docs_grouped_by_src_file[file_name] = {
                    "author": doc.metadata["author"],
                    "url": doc.metadata["url"],
                    "title": (
                        doc.metadata["title"]
                        .replace("{", "")
                        .replace("}", "")
                    ),
                    "texts": [],
                }
            docs_grouped_by_src_file[file_name]["texts"].append({
                "original_text": (
                    doc.metadata["original_text"]
                    .strip()
                    .replace("\n", " ")
                    .replace('"', "'")
                ),
                "conf": confirmation_by_document[doc.uid],
                "conf_level":
                    confirmation_level(
                        confirmation_by_document[doc.uid]
                    ).value,
                "full_text": (
                    doc.text
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
    evse_result: EvidenceSeekerResult,
    translations: dict[str, str] = dict(),
    jinja2_md_template: str | Template | None = None,
    group_docs_by_sources: bool = False,
    show_documents: bool = True,
    **kwargs,
) -> str:
    # TODO: see task from `get_grouped_sources`
    # also: `claims` is, depending from `group_docs_by_sources`
    # differently structured! (refactor)
    if group_docs_by_sources:
        claims = [
            (
                claim,
                get_grouped_sources(
                    claim.documents,
                    claim.confirmation_by_document
                )
            )
            for claim in evse_result.claims
        ]
    else:
        claims = evse_result.claims
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
        claims=claims,
        translation=translations,
        show_documents=show_documents,
        **kwargs,
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
    result_dir: str = "",
    local_base: str = ".",
    subdirectory_construction: str | None = None,
    write_on_github: bool = False,
    github_token_name: str | None = None,
    repo_name: str | None = None,
    additional_markdown_log: bool = False,
    filename_without_suffix: Callable[[EvidenceSeekerResult], str] | str | None = None,
):
    # Do not log results if pipeline failed somehow
    # TODO: Better to use state field (in 'EvSeResult') by catching
    # errors and/or accessing the error codes from request
    # (refactor workflows or pipeline for this)
    if len(evse_result.claims) == 0:
        return
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

