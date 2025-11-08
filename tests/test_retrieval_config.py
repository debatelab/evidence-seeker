"""Tests for retrieval configuration loading and instantiation.

This module tests that the configuration system works correctly with
YAML defaults and allows partial instantiation.
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from evidence_seeker.retrieval.config import (
    RetrievalConfig,
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
        assert "embed_backend_type" in config_dict
        assert "embed_model_name" in config_dict
        assert "top_k" in config_dict
        assert "window_size" in config_dict

    def test_get_default_for_field(self):
        """Test that individual field defaults can be retrieved."""
        embed_backend_type = _get_default_for_field("embed_backend_type")
        assert embed_backend_type == "huggingface"
        
        embed_model_name = _get_default_for_field("embed_model_name")
        expected_model = (
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        assert embed_model_name == expected_model
        
        top_k = _get_default_for_field("top_k")
        assert top_k == 8
        
        window_size = _get_default_for_field("window_size")
        assert window_size == 3
        
        embed_batch_size = _get_default_for_field("embed_batch_size")
        assert embed_batch_size == 32


class TestPartialInstantiation:
    """Test that configs can be instantiated with only required fields."""

    def test_minimal_instantiation(self):
        """Test instantiation with no required fields."""
        config = RetrievalConfig()
        
        # Check that defaults from YAML are loaded
        assert config.config_version == "v0.1"
        assert config.embed_backend_type == "huggingface"
        expected_model = (
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        assert config.embed_model_name == expected_model
        assert config.top_k == 8
        assert config.window_size == 3
        assert config.embed_batch_size == 32
        assert config.embed_base_url is None

    def test_partial_override(self):
        """Test instantiation with some fields overridden."""
        config = RetrievalConfig(
            embed_backend_type="tei",
            top_k=10,
            window_size=5,
            embed_model_name="custom-model",
            embed_base_url="http://localhost:8080"
        )
        
        # Check overridden fields
        assert config.embed_backend_type == "tei"
        assert config.top_k == 10
        assert config.window_size == 5
        assert config.embed_model_name == "custom-model"
        
        # Check that other defaults are still loaded
        assert config.config_version == "v0.1"
        assert config.embed_batch_size == 32

    def test_custom_paths(self):
        """Test overriding path fields."""
        custom_index_path = "path/to/custom/embeddings"
        custom_doc_dir = "path/to/custom/documents"
        
        config = RetrievalConfig(
            index_persist_path=custom_index_path,
            document_input_dir=custom_doc_dir
        )
        
        assert config.index_persist_path == custom_index_path
        assert config.document_input_dir == custom_doc_dir
        # Other defaults should still be loaded
        assert config.embed_backend_type == "huggingface"
        assert config.top_k == 8


class TestFromYaml:
    """Test loading complete configuration from YAML files."""

    def test_from_yaml_package_data(self):
        """Test loading from the package data YAML file."""
        # Use importlib.resources to access package data
        import importlib.resources as pkg_resources
        
        try:
            config_file = pkg_resources.files(
                "evidence_seeker.package_data"
            ).joinpath("config/retrieval_config.yaml")
            
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
            config = RetrievalConfig.from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink(missing_ok=True)
        
        assert isinstance(config, RetrievalConfig)
        assert config.config_version == "v0.1"
        assert config.embed_backend_type == "huggingface"
        expected_model = (
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        assert config.embed_model_name == expected_model

    def test_from_yaml_custom_file(self):
        """Test loading from a custom YAML file."""
        # Create a temporary YAML file with custom config
        custom_config = {
            "config_version": "v0.2",
            "description": "Custom retrieval config",
            "embed_backend_type": "tei",
            "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embed_base_url": "http://localhost:8080",
            "index_persist_path": "custom/embeddings",
            "document_input_dir": "custom/documents",
            "top_k": 15,
            "window_size": 4,
            "embed_batch_size": 64,
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            config = RetrievalConfig.from_yaml(temp_path)
            
            assert config.config_version == "v0.2"
            assert config.description == "Custom retrieval config"
            assert config.embed_backend_type == "tei"
            assert config.embed_model_name == (
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            assert config.embed_base_url == "http://localhost:8080"
            assert config.index_persist_path == "custom/embeddings"
            assert config.document_input_dir == "custom/documents"
            assert config.top_k == 15
            assert config.window_size == 4
            assert config.embed_batch_size == 64
        finally:
            # Clean up temp file
            Path(temp_path).unlink()

    def test_from_config_file_delegates_to_from_yaml(self):
        """Test that from_config_file properly delegates to from_yaml."""
        # Create a temporary YAML file
        custom_config = {
            "config_version": "v0.1",
            "embed_backend_type": "huggingface",
            "embed_model_name": "custom-model",
            "top_k": 20,
            "index_persist_path": "./custom_embeddings",
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            # Test with string path
            config1 = RetrievalConfig.from_config_file(temp_path)
            assert config1.embed_model_name == "custom-model"
            assert config1.top_k == 20
            
            # Test with Path object converted to string
            config2 = RetrievalConfig.from_config_file(str(Path(temp_path)))
            assert config2.embed_model_name == "custom-model"
            assert config2.top_k == 20
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_partial_fields_minimal(self):
        """Test loading from YAML with minimal subset of fields.
        
        Fields not specified in YAML should use default values,
        while specified fields should have custom values.
        """
        # Only specify a few fields
        partial_config = {
            "top_k": 12,
            "window_size": 5,
            "index_persist_path": "./my_embeddings",
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            config = RetrievalConfig.from_yaml(temp_path)
            
            # Specified fields should have custom values
            assert config.top_k == 12
            assert config.window_size == 5
            assert config.index_persist_path == "./my_embeddings"
            
            # Unspecified fields should have default values from YAML
            assert config.config_version == "v0.1"
            assert config.embed_backend_type == "huggingface"
            expected_model = (
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
            assert config.embed_model_name == expected_model
            assert config.embed_batch_size == 32
            expected_desc = (
                "Configuration of EvidenceSeeker's retriever component."
            )
            assert config.description == expected_desc
            assert config.embed_base_url is None
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_partial_fields_mixed(self):
        """Test loading from YAML with a mixed subset of fields.
        
        Tests that defaults are properly used for unspecified fields
        while respecting custom values for specified fields.
        """
        # Specify various fields across different categories
        partial_config = {
            "embed_backend_type": "tei",
            "embed_base_url": "http://custom-server:9000",
            "description": "Partial config test",
            "embed_batch_size": 128,
            "index_persist_path": "./partial_embeddings",
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            config = RetrievalConfig.from_yaml(temp_path)
            
            # Specified fields should have custom values
            assert config.embed_backend_type == "tei"
            assert config.embed_base_url == "http://custom-server:9000"
            assert config.description == "Partial config test"
            assert config.embed_batch_size == 128
            assert config.index_persist_path == "./partial_embeddings"
            
            # Unspecified fields should have default values
            assert config.config_version == "v0.1"
            expected_model = (
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
            assert config.embed_model_name == expected_model
            assert config.top_k == 8
            assert config.window_size == 3
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_partial_fields_paths_only(self):
        """Test loading from YAML specifying only path-related fields.
        
        All other fields should use defaults.
        """
        partial_config = {
            "index_persist_path": "./test_index",
            "document_input_dir": "./test_docs",
            "meta_data_file": "./test_metadata.json",
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            config = RetrievalConfig.from_yaml(temp_path)
            
            # Specified path fields
            assert config.index_persist_path == "./test_index"
            assert config.document_input_dir == "./test_docs"
            assert config.meta_data_file == "./test_metadata.json"
            
            # All other fields should be defaults
            assert config.config_version == "v0.1"
            assert config.embed_backend_type == "huggingface"
            expected_model = (
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
            assert config.embed_model_name == expected_model
            assert config.top_k == 8
            assert config.window_size == 3
            assert config.embed_batch_size == 32
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_empty_file_uses_all_defaults(self):
        """Test that an empty YAML file results in all default values."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            # Write empty YAML (or just empty dict)
            yaml.dump({}, f)
            temp_path = f.name
        
        try:
            config = RetrievalConfig.from_yaml(temp_path)
            
            # All fields should have default values
            assert config.config_version == "v0.1"
            assert config.embed_backend_type == "huggingface"
            expected_model = (
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
            assert config.embed_model_name == expected_model
            assert config.top_k == 8
            assert config.window_size == 3
            assert config.embed_batch_size == 32
            expected_desc = (
                "Configuration of EvidenceSeeker's retriever component."
            )
            assert config.description == expected_desc
        finally:
            Path(temp_path).unlink()


class TestEmbedBackendTypeValidation:
    """Test embed backend type validation."""

    def test_valid_embed_backend_types(self):
        """Test that valid embed backend types are accepted."""
        for backend_type in ["huggingface", "tei", "huggingface_inference_api"]:
            if backend_type in ["tei", "huggingface_inference_api"]:
                # These require embed_base_url
                config = RetrievalConfig(
                    embed_backend_type=backend_type,
                    embed_base_url="http://localhost:8080"
                )
            else:
                config = RetrievalConfig(
                    embed_backend_type=backend_type,
                    index_persist_path="./embeddings"
                )
            assert config.embed_backend_type == backend_type

    def test_tei_requires_base_url(self):
        """Test that TEI backend requires embed_base_url."""
        with pytest.raises(Exception):  # ValueError
            RetrievalConfig(
                embed_backend_type="tei",
                index_persist_path="./embeddings"
            )


class TestParameterRanges:
    """Test that numeric parameters have sensible ranges."""

    def test_top_k_positive(self):
        """Test that top_k must be positive."""
        config = RetrievalConfig(
            top_k=10,
            index_persist_path="./embeddings"
        )
        assert config.top_k == 10

    def test_window_size_non_negative(self):
        """Test that window_size can be 0 or positive."""
        config1 = RetrievalConfig(
            window_size=0,
            index_persist_path="./embeddings"
        )
        assert config1.window_size == 0
        
        config2 = RetrievalConfig(
            window_size=5,
            index_persist_path="./embeddings"
        )
        assert config2.window_size == 5

    def test_embed_batch_size_positive(self):
        """Test that embed_batch_size must be positive."""
        config = RetrievalConfig(
            embed_batch_size=64,
            index_persist_path="./embeddings"
        )
        assert config.embed_batch_size == 64


class TestOptionalFields:
    """Test that optional fields work correctly."""

    def test_embed_base_url_optional(self):
        """Test that embed_base_url is optional."""
        config1 = RetrievalConfig(index_persist_path="./embeddings")
        assert config1.embed_base_url is None
        
        config2 = RetrievalConfig(
            embed_base_url="http://localhost:8080",
            embed_backend_type="tei",
            index_persist_path="./embeddings"
        )
        assert config2.embed_base_url == "http://localhost:8080"

    def test_description_has_default(self):
        """Test that description has a default value from YAML."""
        config1 = RetrievalConfig(index_persist_path="./embeddings")
        expected_desc = "Configuration of EvidenceSeeker's retriever component."
        assert config1.description == expected_desc
        
        config2 = RetrievalConfig(
            description="My custom config",
            index_persist_path="./embeddings"
        )
        assert config2.description == "My custom config"


class TestBackwardCompatibility:
    """Test backward compatibility with old config loading methods."""

    def test_from_config_file_still_works(self):
        """Test that from_config_file method still works (backward compat)."""
        custom_config = {
            "config_version": "v0.1",
            "embed_backend_type": "huggingface",
            "top_k": 7,
            "index_persist_path": "./embeddings",
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            config = RetrievalConfig.from_config_file(temp_path)
            assert isinstance(config, RetrievalConfig)
            assert config.embed_backend_type == "huggingface"
            assert config.top_k == 7
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
