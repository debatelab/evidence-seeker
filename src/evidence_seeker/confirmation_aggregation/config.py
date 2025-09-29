import pydantic 

class ConfirmationAggregationConfig(pydantic.BaseModel):
    confirmation_threshold: float = 0.2