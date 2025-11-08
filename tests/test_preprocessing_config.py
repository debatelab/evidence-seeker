"""Tests for preprocessing configuration loading and instantiation.

This module tests that the configuration system works correctly with
YAML defaults and allows partial instantiation.
"""

import pytest
from pathlib import Path
import tempfile
import yaml


from evidence_seeker.preprocessing.config import (
    ClaimPreprocessingConfig,
    PreprocessorStepConfig,
    PreprocessorModelStepConfig,
    _load_default_config_dict,
    _get_default_for_field,
)


class TestConfigLoading:
    """Test YAML configuration loading and defaults."""

    def test_load_default_config_dict(self):
        """Test that default config dict can be loaded from YAML."""
        config_dict = _load_default_config_dict()
        
        assert isinstance(config_dict, dict)
        assert "config_version" in config_dict
        assert "system_prompt" in config_dict
        assert "language" in config_dict
        assert "timeout" in config_dict
        assert "verbose" in config_dict
        assert "freetext_descriptive_analysis" in config_dict
        assert "list_descriptive_statements" in config_dict

    def test_get_default_for_field(self):
        """Test that individual field defaults can be retrieved."""
        system_prompt = _get_default_for_field("system_prompt")
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "critical thinking" in system_prompt.lower()
        
        language = _get_default_for_field("language")
        assert language == "DE"
        
        timeout = _get_default_for_field("timeout")
        assert timeout == 900

    def test_get_default_for_step_config(self):
        """Test that step configurations can be retrieved from defaults."""
        step_config = _get_default_for_field("freetext_descriptive_analysis")
        
        assert isinstance(step_config, dict)
        assert step_config["name"] == "freetext_descriptive_analysis"
        assert "description" in step_config
        assert "llm_specific_configs" in step_config
        assert "default" in step_config["llm_specific_configs"]


class TestPartialInstantiation:
    """Test that configs can be instantiated with only required fields."""

    def test_minimal_instantiation(self):
        """Test instantiation with only the required field."""
        config = ClaimPreprocessingConfig(used_model_key="test_model")
        
        # Check required field
        assert config.used_model_key == "test_model"
        
        # Check that defaults from YAML are loaded
        assert config.config_version == "v0.1"
        assert "critical thinking" in config.system_prompt.lower()
        assert config.language == "DE"
        assert config.timeout == 900
        assert config.verbose is False
        
        # Check step configurations are loaded
        assert (config.freetext_descriptive_analysis.name ==
                "freetext_descriptive_analysis")
        assert (config.list_descriptive_statements.name ==
                "list_descriptive_statements")
        assert (config.freetext_ascriptive_analysis.name ==
                "freetext_ascriptive_analysis")
        assert (config.list_ascriptive_statements.name ==
                "list_ascriptive_statements")
        assert (config.freetext_normative_analysis.name ==
                "freetext_normative_analysis")
        assert (config.list_normative_statements.name ==
                "list_normative_statements")
        assert config.negate_claim.name == "negate_claim"

    def test_partial_override(self):
        """Test instantiation with some fields overridden."""
        config = ClaimPreprocessingConfig(
            used_model_key="my_model",
            language="EN",
            timeout=600,
            verbose=True
        )
        
        # Check overridden fields
        assert config.used_model_key == "my_model"
        assert config.language == "EN"
        assert config.timeout == 600
        assert config.verbose is True
        
        # Check that other defaults are still loaded
        assert config.config_version == "v0.1"
        assert "critical thinking" in config.system_prompt.lower()
        assert (config.freetext_descriptive_analysis.name ==
                "freetext_descriptive_analysis")

    def test_custom_system_prompt(self):
        """Test overriding the system prompt."""
        custom_prompt = "You are a custom assistant."
        config = ClaimPreprocessingConfig(
            used_model_key="test_model",
            system_prompt=custom_prompt
        )
        
        assert config.system_prompt == custom_prompt
        # Other defaults should still be loaded
        assert config.language == "DE"
        assert config.timeout == 900


class TestFromYaml:
    """Test loading complete configuration from YAML files."""

    def test_from_yaml_package_data(self):
        """Test loading from the package data YAML file."""
        # Use importlib.resources to access package data
        import importlib.resources as pkg_resources
        
        try:
            config_file = pkg_resources.files(
                "evidence_seeker.package_data"
            ).joinpath("config/preprocessing_config.yaml")
            
            # Convert to string path for from_yaml
            with config_file.open('r') as f:
                # Write to temp file to test from_yaml
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.yaml', delete=False
                ) as tmp:
                    tmp.write(f.read())
                    yaml_path = tmp.name
        except (FileNotFoundError, AttributeError):
            pytest.skip("Package data YAML not found")
        
        try:
            config = ClaimPreprocessingConfig.from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink(missing_ok=True)
        
        assert isinstance(config, ClaimPreprocessingConfig)
        assert config.config_version == "v0.1"
        assert config.used_model_key == "key_model_one"
        assert config.language == "DE"
        assert len(config.models) > 0

    def test_from_yaml_custom_file(self):
        """Test loading from a custom YAML file."""
        # Create a temporary YAML file with custom config
        custom_config = {
            "config_version": "v0.2",
            "description": "Custom test config",
            "system_prompt": "Custom system prompt",
            "language": "EN",
            "timeout": 300,
            "verbose": True,
            "used_model_key": "custom_model",
            "models": {
                "custom_model": {
                    "name": "test_model",
                    "backend_type": "openai",
                    "model": "gpt-4",
                    "api_key_name": "TEST_API_KEY",
                    "temperature": 0.5,
                }
            },
            "freetext_descriptive_analysis": {
                "name": "freetext_descriptive_analysis",
                "description": "Custom description",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "Custom prompt: {claim}"
                    }
                }
            },
            "list_descriptive_statements": {
                "name": "list_descriptive_statements",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "List: {claim}",
                        "guidance_type": "pydantic"
                    }
                }
            },
            "freetext_ascriptive_analysis": {
                "name": "freetext_ascriptive_analysis",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "Ascriptive: {claim}"
                    }
                }
            },
            "list_ascriptive_statements": {
                "name": "list_ascriptive_statements",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "List ascriptive: {claim}",
                        "guidance_type": "pydantic"
                    }
                }
            },
            "freetext_normative_analysis": {
                "name": "freetext_normative_analysis",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "Normative: {claim}"
                    }
                }
            },
            "list_normative_statements": {
                "name": "list_normative_statements",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "List normative: {claim}",
                        "guidance_type": "pydantic"
                    }
                }
            },
            "negate_claim": {
                "name": "negate_claim",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "Negate: {statement}"
                    }
                }
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            config = ClaimPreprocessingConfig.from_yaml(temp_path)
            
            assert config.config_version == "v0.2"
            assert config.description == "Custom test config"
            assert config.system_prompt == "Custom system prompt"
            assert config.language == "EN"
            assert config.timeout == 300
            assert config.verbose is True
            assert config.used_model_key == "custom_model"
            assert "custom_model" in config.models
            
            # Check step config
            step_config = config.freetext_descriptive_analysis
            assert step_config.description == "Custom description"
            model_config = step_config.llm_specific_configs["default"]
            assert model_config.prompt_template == "Custom prompt: {claim}"
        finally:
            # Clean up temp file
            Path(temp_path).unlink()

    def test_from_yaml_partial_fields_minimal(self):
        """Test loading from YAML with minimal subset of fields.
        
        Fields not specified in YAML should use default values,
        while specified fields should have custom values.
        """
        # Only specify required field and a few optional ones
        partial_config = {
            "used_model_key": "my_test_model",
            "language": "EN",
            "timeout": 600,
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            config = ClaimPreprocessingConfig.from_yaml(temp_path)
            
            # Specified fields should have custom values
            assert config.used_model_key == "my_test_model"
            assert config.language == "EN"
            assert config.timeout == 600
            
            # Unspecified fields should have default values
            assert config.config_version == "v0.1"
            assert "critical thinking" in config.system_prompt.lower()
            assert config.verbose is False
            expected_desc = (
                "Configuration of EvidenceSeeker's preprocessing component."
            )
            assert config.description == expected_desc
            
            # Step configs should be defaults
            assert (config.freetext_descriptive_analysis.name ==
                    "freetext_descriptive_analysis")
            assert (config.list_descriptive_statements.name ==
                    "list_descriptive_statements")
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_partial_fields_with_one_step(self):
        """Test loading from YAML with required field and one custom step.
        
        Custom step should be loaded, all other steps should use defaults.
        """
        partial_config = {
            "used_model_key": "custom_key",
            "verbose": True,
            "freetext_descriptive_analysis": {
                "name": "freetext_descriptive_analysis",
                "description": "My custom step",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "Analyze this: {claim}"
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            config = ClaimPreprocessingConfig.from_yaml(temp_path)
            
            # Specified fields
            assert config.used_model_key == "custom_key"
            assert config.verbose is True
            
            # Custom step
            custom_step = config.freetext_descriptive_analysis
            assert custom_step.description == "My custom step"
            assert (custom_step.llm_specific_configs["default"].prompt_template
                    == "Analyze this: {claim}")
            
            # Other steps should use defaults
            default_step = config.list_descriptive_statements
            assert default_step.name == "list_descriptive_statements"
            # Should have default prompt template (not the custom one)
            assert "Analyze this" not in (
                default_step.llm_specific_configs["default"].prompt_template
            )
            
            # Unspecified simple fields should use defaults
            assert config.config_version == "v0.1"
            assert config.language == "DE"
            assert config.timeout == 900
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_partial_fields_system_prompt_only(self):
        """Test loading from YAML with custom system_prompt only.
        
        All other fields should use defaults.
        """
        partial_config = {
            "used_model_key": "test_key",
            "system_prompt": "You are a specialized fact-checker.",
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            config = ClaimPreprocessingConfig.from_yaml(temp_path)
            
            # Custom system prompt
            assert config.system_prompt == "You are a specialized fact-checker."
            
            # All other fields should be defaults
            assert config.config_version == "v0.1"
            assert config.language == "DE"
            assert config.timeout == 900
            assert config.verbose is False
            expected_desc = (
                "Configuration of EvidenceSeeker's preprocessing component."
            )
            assert config.description == expected_desc
            
            # All step configs should be defaults
            assert (config.freetext_descriptive_analysis.name ==
                    "freetext_descriptive_analysis")
            assert (config.negate_claim.name == "negate_claim")
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_partial_models_field(self):
        """Test loading from YAML with partial models configuration.
        
        Only specified models should be in config, with custom values.
        """
        partial_config = {
            "used_model_key": "my_model",
            "models": {
                "my_model": {
                    "name": "test_llm",
                    "backend_type": "openai",
                    "model": "gpt-3.5-turbo",
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            config = ClaimPreprocessingConfig.from_yaml(temp_path)
            
            # Models field should have exactly what we specified
            assert "my_model" in config.models
            assert config.models["my_model"]["name"] == "test_llm"
            assert config.models["my_model"]["backend_type"] == "openai"
            assert len(config.models) == 1
            
            # All other fields should be defaults
            assert config.config_version == "v0.1"
            assert config.language == "DE"
            assert config.timeout == 900
        finally:
            Path(temp_path).unlink()


class TestStepConfigurations:
    """Test that step configurations work correctly."""

    def test_all_steps_have_default_configs(self):
        """Test that all steps have default llm_specific_configs."""
        config = ClaimPreprocessingConfig(used_model_key="test_model")
        
        steps = [
            config.freetext_descriptive_analysis,
            config.list_descriptive_statements,
            config.freetext_ascriptive_analysis,
            config.list_ascriptive_statements,
            config.freetext_normative_analysis,
            config.list_normative_statements,
            config.negate_claim,
        ]
        
        for step in steps:
            assert isinstance(step, PreprocessorStepConfig)
            assert "default" in step.llm_specific_configs
            default_config = step.llm_specific_configs["default"]
            assert isinstance(default_config, PreprocessorModelStepConfig)
            assert len(default_config.prompt_template) > 0

    def test_guidance_type_validation(self):
        """Test that guidance_type is validated correctly."""
        config = ClaimPreprocessingConfig(used_model_key="test_model")
        
        # Check that list steps have guidance_type set to "json"
        list_desc = config.list_descriptive_statements
        assert list_desc.llm_specific_configs["default"].guidance_type == "pydantic"
        list_asc = config.list_ascriptive_statements
        assert list_asc.llm_specific_configs["default"].guidance_type == "pydantic"
        list_norm = config.list_normative_statements
        assert list_norm.llm_specific_configs["default"].guidance_type == "pydantic"
        
        # Check that freetext steps don't have guidance_type
        free_desc = config.freetext_descriptive_analysis
        assert free_desc.llm_specific_configs["default"].guidance_type is None
        neg = config.negate_claim
        assert neg.llm_specific_configs["default"].guidance_type is None

    def test_get_step_config_method(self):
        """Test the get_step_config helper method."""
        config = ClaimPreprocessingConfig(used_model_key="test_model")
        
        # Test by step name
        step_name = "freetext_descriptive_analysis"
        step_config = config.get_step_config(step_name=step_name)
        assert isinstance(step_config, PreprocessorModelStepConfig)
        assert "claim" in step_config.prompt_template
        
        # Test by step config object
        list_step = config.list_descriptive_statements
        step_config = config.get_step_config(step_config=list_step)
        assert isinstance(step_config, PreprocessorModelStepConfig)
        assert step_config.guidance_type == "pydantic"

    def test_get_chat_template_method(self):
        """Test the get_chat_template helper method."""
        config = ClaimPreprocessingConfig(used_model_key="test_model")
        
        step_name = "freetext_descriptive_analysis"
        chat_template = config.get_chat_template(step_name=step_name)
        assert chat_template is not None

    def test_get_system_prompt_method(self):
        """Test the get_system_prompt helper method."""
        config = ClaimPreprocessingConfig(used_model_key="test_model")
        
        step_name = "freetext_descriptive_analysis"
        system_prompt = config.get_system_prompt(step_name=step_name)
        assert isinstance(system_prompt, str)
        assert "critical thinking" in system_prompt.lower()


class TestValidation:
    """Test configuration validation."""

    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises validation error."""
        with pytest.raises(Exception):  # pydantic.ValidationError
            # Missing required field: used_model_key
            ClaimPreprocessingConfig()  # type: ignore

    def test_invalid_guidance_type_raises_error(self):
        """Test that invalid guidance_type raises validation error."""
        with pytest.raises(Exception):  # pydantic.ValidationError
            PreprocessorModelStepConfig(
                prompt_template="test",
                guidance_type="invalid_type"
            )

    # def test_valid_guidance_type_accepted(self):
    #     """Test that valid guidance_type is accepted."""
    #     model_config = PreprocessorModelStepConfig(
    #         prompt_template="test",
    #         guidance_type="json"
    #     )
    #     assert model_config.guidance_type == "json"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
