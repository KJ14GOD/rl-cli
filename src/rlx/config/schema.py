from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
Probability = Annotated[float, Field(ge=0, le=1)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class RLXModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvConfig(RLXModel):
    id: NonEmptyStr
    num_envs: PositiveInt
    import_modules: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("import_modules")
    @classmethod
    def validate_import_modules(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class AlgoConfig(RLXModel):
    name: Literal["ppo"]
    total_timesteps: PositiveInt
    rollout_steps: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    gamma: Probability
    gae_lambda: Probability
    clip_range: NonNegativeFloat
    entropy_coef: NonNegativeFloat
    value_coef: NonNegativeFloat
    update_epochs: PositiveInt


class PolicyConfig(RLXModel):
    type: Literal["mlp", "custom"]
    hidden_sizes: list[PositiveInt] | None = None
    import_module: NonEmptyStr | None = None
    class_name: NonEmptyStr | None = None

    @field_validator("hidden_sizes")
    @classmethod
    def validate_hidden_sizes(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("must contain at least one layer size")
        return value

    @model_validator(mode="after")
    def validate_policy_shape(self) -> "PolicyConfig":
        if self.type == "mlp":
            if not self.hidden_sizes:
                raise ValueError("hidden_sizes is required when policy.type is 'mlp'")
            if self.import_module is not None or self.class_name is not None:
                raise ValueError(
                    "import_module and class_name are only supported when policy.type is 'custom'"
                )
            return self

        if self.import_module is None or self.class_name is None:
            raise ValueError(
                "import_module and class_name are required when policy.type is 'custom'"
            )
        if self.hidden_sizes is not None:
            raise ValueError("hidden_sizes is only supported when policy.type is 'mlp'")
        return self


class CheckpointConfig(RLXModel):
    save_every: PositiveInt


class EvalConfig(RLXModel):
    every: PositiveInt
    episodes: PositiveInt
    deterministic: bool


class ExperimentConfig(RLXModel):
    run_name: NonEmptyStr
    seed: NonNegativeInt
    device: NonEmptyStr
    env: EnvConfig
    algo: AlgoConfig
    policy: PolicyConfig
    checkpoint: CheckpointConfig
    eval: EvalConfig
