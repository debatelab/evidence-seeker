"""Tests for confirmation analysis configuration loading and instantiation.

This module tests that the configuration system works correctly with
YAML defaults and allows partial instantiation.
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from evidence_seeker.confirmation_analysis.config import (
    ConfirmationAnalyzerConfig,
    ConfirmationAnalyzerStepConfig,
    ConfirmationAnalyzerModelStepConfig,
    MultipleChoiceTaskStepConfig,
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
        assert "timeout" in config_dict
        assert "verbose" in config_dict
        assert "freetext_confirmation_analysis" in config_dict
        assert "multiple_choice_confirmation_analysis" in config_dict

    def test_get_default_for_field(self):
        """Test that individual field defaults can be retrieved."""
        system_prompt = _get_default_for_field("system_prompt")
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "critical thinking" in system_prompt.lower()
        
        timeout = _get_default_for_field("timeout")
        assert timeout == 900
        
        verbose = _get_default_for_field("verbose")
        assert verbose is False

    def test_get_default_for_step_config(self):
        """Test that step configurations can be retrieved from defaults."""
        step_config = _get_default_for_field("freetext_confirmation_analysis")
        
        assert isinstance(step_config, dict)
        assert step_config["name"] == "freetext_confirmation_analysis"
        assert "description" in step_config
        assert "llm_specific_configs" in step_config
        assert "default" in step_config["llm_specific_configs"]


class TestPartialInstantiation:
    """Test that configs can be instantiated with only required fields."""

    def test_minimal_instantiation(self):
        """Test instantiation with no required fields."""
        config = ConfirmationAnalyzerConfig()
        
        # Check that defaults from YAML are loaded
        assert config.config_version == "v0.2"
        assert "critical thinking" in config.system_prompt.lower()
        assert config.timeout == 900
        assert config.verbose is False
        
        # Check step configurations are loaded
        assert (config.freetext_confirmation_analysis.name ==
                "freetext_confirmation_analysis")
        assert (config.multiple_choice_confirmation_analysis.name ==
                "multiple_choice_confirmation_analysis")

    def test_partial_override(self):
        """Test instantiation with some fields overridden."""
        config = ConfirmationAnalyzerConfig(
            used_model_key="my_model",
            timeout=600,
            verbose=True
        )
        
        # Check overridden fields
        assert config.used_model_key == "my_model"
        assert config.timeout == 600
        assert config.verbose is True
        
        # Check that other defaults are still loaded
        assert config.config_version == "v0.2"
        assert "critical thinking" in config.system_prompt.lower()
        assert (config.freetext_confirmation_analysis.name ==
                "freetext_confirmation_analysis")

    def test_custom_system_prompt(self):
        """Test overriding the system prompt."""
        custom_prompt = "You are a custom assistant."
        config = ConfirmationAnalyzerConfig(
            system_prompt=custom_prompt
        )
        
        assert config.system_prompt == custom_prompt
        # Other defaults should still be loaded
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
            ).joinpath("config/confirmation_analysis_config.yaml")
            
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
            config = ConfirmationAnalyzerConfig.from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink(missing_ok=True)
        
        assert isinstance(config, ConfirmationAnalyzerConfig)
        assert config.config_version == "v0.2"
        assert config.used_model_key == "key_model_one"
        assert len(config.models) > 0

    def test_from_yaml_custom_file(self):
        """Test loading from a custom YAML file."""
        # Create a temporary YAML file with custom config
        custom_config = {
            "config_version": "v0.3",
            "description": "Custom test config",
            "system_prompt": "Custom system prompt",
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
            "freetext_confirmation_analysis": {
                "name": "freetext_confirmation_analysis",
                "description": "Custom RTE analysis",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "Custom prompt: {evidence_item}"
                    }
                }
            },
            "multiple_choice_confirmation_analysis": {
                "name": "multiple_choice_confirmation_analysis",
                "description": "Custom MCQ",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "Choose: {answer_options}",
                        "answer_labels": ["A", "B", "C"],
                        "answer_options": ["Yes", "No", "Maybe"],
                        "guidance_type": "json",
                        "n_repetitions_mcq": 1,
                    }
                }
            },
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            config = ConfirmationAnalyzerConfig.from_yaml(temp_path)
            
            assert config.config_version == "v0.3"
            assert config.description == "Custom test config"
            assert config.system_prompt == "Custom system prompt"
            assert config.timeout == 300
            assert config.verbose is True
            assert config.used_model_key == "custom_model"
            assert "custom_model" in config.models
            
            # Check step config
            step_config = config.freetext_confirmation_analysis
            assert step_config.description == "Custom RTE analysis"
            model_config = step_config.llm_specific_configs["default"]
            assert model_config.prompt_template == "Custom prompt: {evidence_item}"
        finally:
            # Clean up temp file
            Path(temp_path).unlink()

    def test_from_yaml_partial_fields_minimal(self):
        """Test loading from YAML with minimal subset of fields.
        
        Fields not specified in YAML should use default values,
        while specified fields should have custom values.
        """
        # Only specify a few optional fields
        partial_config = {
            "timeout": 450,
            "verbose": True,
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            config = ConfirmationAnalyzerConfig.from_yaml(temp_path)
            
            # Specified fields should have custom values
            assert config.timeout == 450
            assert config.verbose is True
            
            # Unspecified fields should have default values
            assert config.config_version == "v0.2"
            assert "critical thinking" in config.system_prompt.lower()
            expected_desc = (
                "Configuration of EvidenceSeeker's "
                "confirmation analyser component."
            )
            assert config.description == expected_desc
            
            # Step configs should be defaults
            assert (config.freetext_confirmation_analysis.name ==
                    "freetext_confirmation_analysis")
            assert (config.multiple_choice_confirmation_analysis.name ==
                    "multiple_choice_confirmation_analysis")
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_partial_fields_with_one_step(self):
        """Test loading from YAML with one custom step.
        
        Custom step should be loaded, other step should use defaults.
        """
        partial_config = {
            "used_model_key": "partial_model",
            "freetext_confirmation_analysis": {
                "name": "freetext_confirmation_analysis",
                "description": "My custom RTE step",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "Analyze: {evidence_item}"
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
            config = ConfirmationAnalyzerConfig.from_yaml(temp_path)
            
            # Specified field
            assert config.used_model_key == "partial_model"
            
            # Custom step
            custom_step = config.freetext_confirmation_analysis
            assert custom_step.description == "My custom RTE step"
            assert (custom_step.llm_specific_configs["default"].prompt_template
                    == "Analyze: {evidence_item}")
            
            # Other step should use default
            default_step = config.multiple_choice_confirmation_analysis
            assert default_step.name == "multiple_choice_confirmation_analysis"
            # Should have default prompt template
            assert "Analyze:" not in (
                default_step.llm_specific_configs["default"].prompt_template
            )
            
            # Unspecified simple fields should use defaults
            assert config.config_version == "v0.2"
            assert config.timeout == 900
            assert config.verbose is False
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_partial_system_prompt_only(self):
        """Test loading from YAML with custom system_prompt only.
        
        All other fields should use defaults.
        """
        partial_config = {
            "system_prompt": "You are an expert in textual entailment.",
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            config = ConfirmationAnalyzerConfig.from_yaml(temp_path)
            
            # Custom system prompt
            assert config.system_prompt == "You are an expert in textual entailment."
            
            # All other fields should be defaults
            assert config.config_version == "v0.2"
            assert config.timeout == 900
            assert config.verbose is False
            expected_desc = (
                "Configuration of EvidenceSeeker's "
                "confirmation analyser component."
            )
            assert config.description == expected_desc
            
            # All step configs should be defaults
            assert (config.freetext_confirmation_analysis.name ==
                    "freetext_confirmation_analysis")
            assert (config.multiple_choice_confirmation_analysis.name ==
                    "multiple_choice_confirmation_analysis")
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_partial_mcq_step_modifications(self):
        """Test loading from YAML with partial MCQ step modifications.
        
        Only specified MCQ fields should be custom, others default.
        """
        partial_config = {
            "multiple_choice_confirmation_analysis": {
                "name": "multiple_choice_confirmation_analysis",
                "llm_specific_configs": {
                    "default": {
                        "prompt_template": "Custom MCQ: {answer_options}",
                        "answer_labels": ["X", "Y", "Z"],
                        "answer_options": ["Strong", "Moderate", "Weak"],
                        "guidance_type": "json",
                        "n_repetitions_mcq": 5,
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
            config = ConfirmationAnalyzerConfig.from_yaml(temp_path)
            
            # MCQ step should have custom values
            mcq_step = config.multiple_choice_confirmation_analysis
            mcq_config = mcq_step.llm_specific_configs["default"]
            assert mcq_config.prompt_template == "Custom MCQ: {answer_options}"
            assert mcq_config.answer_labels == ["X", "Y", "Z"]
            assert mcq_config.answer_options == ["Strong", "Moderate", "Weak"]
            assert mcq_config.n_repetitions_mcq == 5
            
            # Other step should use default
            freetext_step = config.freetext_confirmation_analysis
            assert freetext_step.name == "freetext_confirmation_analysis"
            
            # All simple fields should be defaults
            assert config.config_version == "v0.2"
            assert config.timeout == 900
            assert config.verbose is False
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_empty_file_uses_all_defaults(self):
        """Test that an empty YAML file results in all default values."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            # Write empty YAML (empty dict)
            yaml.dump({}, f)
            temp_path = f.name
        
        try:
            config = ConfirmationAnalyzerConfig.from_yaml(temp_path)
            
            # All fields should have default values
            assert config.config_version == "v0.2"
            assert "critical thinking" in config.system_prompt.lower()
            assert config.timeout == 900
            assert config.verbose is False
            expected_desc = (
                "Configuration of EvidenceSeeker's "
                "confirmation analyser component."
            )
            assert config.description == expected_desc
            
            # Step configs should be defaults
            assert (config.freetext_confirmation_analysis.name ==
                    "freetext_confirmation_analysis")
            assert (config.multiple_choice_confirmation_analysis.name ==
                    "multiple_choice_confirmation_analysis")
        finally:
            Path(temp_path).unlink()


class TestStepConfigurations:
    """Test that step configurations work correctly."""

    def test_all_steps_have_default_configs(self):
        """Test that all steps have default llm_specific_configs."""
        config = ConfirmationAnalyzerConfig()
        
        steps = [
            config.freetext_confirmation_analysis,
            config.multiple_choice_confirmation_analysis,
        ]
        
        for step in steps:
            assert isinstance(step, ConfirmationAnalyzerStepConfig)
            assert "default" in step.llm_specific_configs
            default_config = step.llm_specific_configs["default"]
            assert isinstance(
                default_config,
                (ConfirmationAnalyzerModelStepConfig, MultipleChoiceTaskStepConfig)
            )
            assert len(default_config.prompt_template) > 0

    def test_multiple_choice_config_structure(self):
        """Test that multiple choice config has expected structure."""
        config = ConfirmationAnalyzerConfig()
        
        mc_step = config.multiple_choice_confirmation_analysis
        mc_config = mc_step.llm_specific_configs["default"]
        
        assert isinstance(mc_config, MultipleChoiceTaskStepConfig)
        assert mc_config.answer_labels is not None
        assert len(mc_config.answer_labels) > 0
        assert mc_config.answer_options is not None
        assert len(mc_config.answer_options) > 0
        assert mc_config.guidance_type is not None

    def test_get_step_config_method(self):
        """Test the get_step_config helper method."""
        config = ConfirmationAnalyzerConfig()
        
        # Test by step name
        step_config = config.get_step_config(
            step_name="freetext_confirmation_analysis"
        )
        assert isinstance(step_config, ConfirmationAnalyzerModelStepConfig)
        assert "TEXT" in step_config.prompt_template
        
        # Test by step config object
        step_config = config.get_step_config(
            step_config=config.multiple_choice_confirmation_analysis
        )
        assert isinstance(step_config, MultipleChoiceTaskStepConfig)

    def test_get_chat_template_method(self):
        """Test the get_chat_template helper method."""
        config = ConfirmationAnalyzerConfig()
        
        chat_template = config.get_chat_template(
            step_name="freetext_confirmation_analysis"
        )
        assert chat_template is not None

    def test_get_system_prompt_method(self):
        """Test the get_system_prompt helper method."""
        config = ConfirmationAnalyzerConfig()
        
        step_name = "freetext_confirmation_analysis"
        system_prompt = config.get_system_prompt(step_name=step_name)
        assert isinstance(system_prompt, str)
        assert "critical thinking" in system_prompt.lower()


class TestValidation:
    """Test configuration validation."""

    def test_valid_config_accepted(self):
        """Test that valid config is accepted."""
        config = ConfirmationAnalyzerConfig(
            used_model_key="test_model",
            timeout=600
        )
        assert config.used_model_key == "test_model"
        assert config.timeout == 600

    def test_multiple_choice_answer_labels_validation(self):
        """Test that multiple choice answer labels are validated."""
        # Valid labels should work
        config = MultipleChoiceTaskStepConfig(
            prompt_template="test",
            answer_labels=["A", "B", "C"],
            guidance_type="json"
        )
        assert config.answer_labels == ["A", "B", "C"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
