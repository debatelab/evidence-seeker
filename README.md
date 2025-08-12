# 🕵️‍♀️ EvidenceSeeker Boilerplate <!-- omit in toc -->

A code template for building AI based apps that fact-check statements against a given knowledge base. 

---
<div align="center">
  <p align="center">
    📙 <a href="https://debatelab.github.io/evidence-seeker">EvidenseSeeker Docs</a>
    🤗 <a href="https://huggingface.co/spaces/DebateLabKIT/evidence-seeker-demo">Hugging Face Demoapp</a>
    <img src="./docs_src/img/logoKIdeKu.jpg" alt="KIdeKu Logo" width="15" style="vertical-align: middle;"> <a href="https://compphil2mmae.github.io/research/kideku/">KIdeKu Project</a>
  </p>
</div>
<br/>


## 🚀 Getting started

There a several ways to set up and run an EvidenceSeeker based on our Boilerplate. For details, see the [official documentation](https://debatelab.github.io/evidence-seeker).

## 💡 The EvidenceSeeker Workflow

The *EvidenceSeeker Pipeline* is based on Large Language Models (LLMs) and proceeds as follows when fact-checking a statement against a knowldge base:

1. In a first step, the evidence seeker identifies different interpretations of an input statement and distinguishes between *descriptive*, *ascriptive*, and *normative* statements.
2. For the descriptive and ascriptive interpretations, the evidence seeker searches for relevant text passages in a given knowledge base and analyzes for each found interpretation the extent to which each text passage confirms or refutes the  interpretation.
3. These individual analyses are aggregated into one of the following confirmation levels for each interpretation :
    + ‘highly confirmed’,
    + ‘confirmed’,
    + ‘weakly confirmed’,
    + ‘neither confirmed nor refuted’,
    + ‘weakly refuted’,
    + ‘refuted’, and
    + ‘highly refuted’.

You can find more information about the pipeline [here](https://debatelab.github.io/evidence-seeker/workflow.html).


## 🙏 Acknowledgements


###  🛠️ Used third-party tools

 *EvidenceSeeker Boilerplate* would not be possible without the wonderful open source community and relies on the following libraries, amongst others:

- **[LLamaIndex](https://docs.llamaindex.ai/en/stable/)** for the implementation of the workflow.
- **[Pydantic](https://pydantic.dev/)** for modeling the configs.

All other dependencies can be found in the `pyproject.toml` file in our  [GitHub repo](https://github.com/debatelab/evidence-seeker).


### 🤝 Collaborations

We presented the project at the [politechathon workshop](https://www.wahlexe.de/en/) in December 2024 and received very constructive feedback.


### 🏛️ Funding 

KIdeKu is funded by the *Federal Ministry of Education, Family Affairs, Senior Citizens, Woman and Youth ([BMBFSFJ](https://www.bmbfsfj.bund.de/bmbfsfj/meta/en))*.


<a href="https://www.bmbfsfj.bund.de/bmbfsfj/meta/en">
  <img src="./docs_src/img/funding.png" alt="BMFSFJ Funding" width="40%">
</a>

## 📄 License

*EvidenceSeeker Boilerplate* is licensed under the [MIT License](https://opensource.org/licenses/MIT).

