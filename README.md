# 🕵️‍♀️ EvidenceSeeker Boilerplate <!-- omit in toc -->

<div align="center">
  <p align="center">
 📖 <a href="https://debatelab.github.io/evidence-seeker">Documentation</a>
 🤗 <a href="https://huggingface.co/spaces/DebateLabKIT/evidence-seeker-demo">Hugging Face Demo App</a>
 📊 <a href="https://debatelab.github.io/evidence-seeker-results/">Example Results</a>
    <img src="./docs_src/img/logoKIdeKu.jpg" alt="KIdeKu Logo" width="15" style="vertical-align: middle;"> <a href="https://compphil2mmae.github.io/research/kideku/">KIdeKu Project</a>
  </p>
</div>
<br/>

A code template for building AI-based apps that fact-check statements against a given knowledge base. 

## 🎯 What is EvidenceSeeker?

EvidenceSeeker Boilerplate is a Python package that provides a fact-checking pipeline with the following steps:

1. **Statement Analysis**: The preprocessor identifies different interpretations of an input statement and categorises them as descriptive, normative, or ascriptive.
2. **Evidence Retrieval**: The retriever searches through your knowledge base for relevant supporting/contradicting evidence.
3. **Confirmation Analysis**: The confimation analyser assesses how well the found evidence supports or refutes claims and aggrated its results by providing confirmation levels for each found interpretation.

<div align="center">
  <p align="center">
  <img src="./docs_src/img/workflow_en.png" alt="Figure of workflow">
  </p>
</div>


## ✨ Key Features

### 🔧 Core Pipeline

- **Multiple AI Backends**: Support for different inference APIs and local models via LlamaIndex
- **Vector Search**: Semantic search through documents using state-of-the-art embeddings
- **Flexible Configuration**: YAML-based configuration for all pipeline components

### 🖥️ Easy-to-Use Interface

- **CLI Tool**: Complete command-line interface (`evse`) for project initialization and pipeline execution
- **Demo Web App**: Ready-to-deploy Gradio app with multilingual support (German/English)
- **Programmatic API**: Import and use EvidenceSeeker directly in your Python projects

### 📊 Knowledge Base Management

- **Document Indexing**: Build searchable vector indexes from your document collections
- **Metadata Support**: Rich metadata handling for document attribution and source tracking
- **Hub Integration**: Upload/download indexes to/from Hugging Face Hub

## 🚀 Quick Start

There are several ways to set up and run an EvidenceSeeker based on our Boilerplate. For details, see the [official documentation](https://debatelab.github.io/evidence-seeker/getting_started.html).

### Installation

```bash
pip install evidence-seeker
```

### Initialize an EvidenceSeeker
```bash
evse init --name my-fact-checker
cd my-fact-checker
```

### Configuration with API keys

See <https://debatelab.github.io/evidence-seeker/configuration.html>.

### Build Knowledge Base Index
```bash
# Add your documents to knowledge_base/data_files/
evse build-index
```

### Run Fact-Checking
```bash
evse run -i "Your statement to fact-check"
```

### Launch Demo App
```bash
evse demo-app
```

## 📦 What's Included

- **Core Library**: Complete fact-checking pipeline with AI-powered analysis
- **CLI Tool**: Command-line interface for all operations
- **Web Demo**: Gradio-based web application with authentication and result persistence
- **Configuration Templates**: Pre-configured YAML files for immediate use
- **Documentation**: Comprehensive guides and API documentation
- **Example Data**: Sample knowledge base and configurations

## 🛠️ Powered By

- **[LlamaIndex](https://docs.llamaindex.ai/)**: Workflow orchestration and document processing
- **[Gradio](https://gradio.app/)**: Interactive web interface
- **[Pydantic](https://pydantic.dev/)**: Data validation and configuration management
- **[Sentence Transformers](https://www.sbert.net/)**: Document embeddings
- **[Hugging Face](https://huggingface.co/)**: Model hosting and deployment

## 🎯 Use Cases

- **Academic Research**: Fact-check claims against scientific literature
- **Journalism**: Verify statements against reliable source databases
- **Policy Analysis**: Check policy claims against government documents
- **Corporate Compliance**: Validate statements against internal documentation
- **Educational Tools**: Create fact-checking exercises with custom knowledge bases

## 💡 The EvidenceSeeker Workflow

The *EvidenceSeeker Pipeline* is based on Large Language Models (LLMs) and proceeds as follows when fact-checking a statement against a knowledge base:

1. In a first step, the evidence seeker identifies different interpretations of an input statement and distinguishes between *descriptive*, *ascriptive*, and *normative* statements.
2. For each of the found descriptive and ascriptive interpretations, the evidence seeker searches for relevant text passages in a given knowledge base and analyses the extent to which each text passage confirms or refutes the interpretation.
3. These individual analyses are aggregated into one of the following confirmation levels for each interpretation :
    + ‘highly confirmed’,
    + ‘confirmed’,
    + ‘weakly confirmed’,
    + ‘neither confirmed nor refuted’,
    + ‘weakly refuted’,
    + ‘refuted’, and
    + ‘highly refuted’.

You can find more information about the pipeline [here](https://debatelab.github.io/evidence-seeker/workflow.html).

## 🐛 Known Limitations

- Current demo uses German political science texts as knowledge base
- API timeouts may occur on resource-constrained deployments
- Large knowledge bases may require significant computational resources

## 📚 Documentation & Links

- **📖 Documentation**: [https://debatelab.github.io/evidence-seeker](https://debatelab.github.io/evidence-seeker)
- **🤗 Demo App**: [https://huggingface.co/spaces/DebateLabKIT/evidence-seeker-demo](https://huggingface.co/spaces/DebateLabKIT/evidence-seeker-demo)
- **📊 Example Results**: [https://debatelab.github.io/evidence-seeker-results/](https://debatelab.github.io/evidence-seeker-results/)
- **🔬 KIdeKu Project**: [https://compphil2mmae.github.io/research/kideku/](https://compphil2mmae.github.io/research/kideku/)


## 🙏 Acknowledgements

### 🤝 Collaborations

We presented the project at the [Politechathon Workshop](https://www.wahlexe.de/en/) in December 2024 and received constructive feedback.

### 🏛️ Funding 

KIdeKu is funded by the *Federal Ministry of Education, Family Affairs, Senior Citizens, Women and Youth ([BMBFSFJ](https://www.bmbfsfj.bund.de/bmbfsfj/meta/en))*.


<a href="https://www.bmbfsfj.bund.de/bmbfsfj/meta/en">
  <img src="./docs_src/img/funding.png" alt="BMFSFJ Funding" width="40%">
</a>

## 📄 License

*EvidenceSeeker Boilerplate* is licensed under the [MIT License](https://opensource.org/licenses/MIT).

