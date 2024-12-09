"preprocessing.py"


from evidence_seeker.models import CheckedClaim
from evidence_seeker.backend import log_msg, get_openai_llm
from evidence_seeker.preprocessing.simple_preprocessing_workflow import SimplePreprocessingWorkflow
from evidence_seeker.preprocessing.preprocessing_separate_listings_workflow import PreprocessingSeparateListingsWorkflow



class ClaimPreprocessor:
    def __init__(self, config: dict, **kwargs):
        # init model
        api_key_name = config["models"][config["used_model"]]["api_key_name"]
        model = config["models"][config["used_model"]]["model"]
        base_url = config["models"][config["used_model"]]["base_url"]

        max_tokens = 2048  # default value
        if "max_tokens" in config["models"][config["used_model"]]:
            max_tokens = config["models"][config["used_model"]]["max_tokens"]

        context_window = 3900
        if "context_window" in config["models"][config["used_model"]]:
            context_window = config["models"][config["used_model"]][
                "context_window"
            ]

        temperature = 0.1
        if "temperature" in config["models"][config["used_model"]]:
            temperature = config["models"][config["used_model"]]["temperature"]

        log_msg(
            f"Init open ai model {model} (from {base_url}) with"
            f"temperature={temperature} max_tokens={max_tokens}"
            f"and context_window={context_window}"
        )

        llm = get_openai_llm(
            api_key_name=api_key_name,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            context_window=context_window,
            temperature=temperature,
        )
        self.workflow = SimplePreprocessingWorkflow(
            config=config,
            llm=llm,
            timeout=config["pipeline"]["preprocessing"]["timeout"],
            verbose=config["pipeline"]["preprocessing"]["verbose"],
        )

    async def __call__(self, claim: str) -> list[CheckedClaim]:
        workflow_result = await self.workflow.run(claim=claim)
        return workflow_result["checked_claims"]

