"""Smoke tests for importability and basic public API."""


def test_import_sigmutsel():
    """Import the top-level package."""
    import sigmutsel  # noqa: F401


def test_public_api_symbols():
    """Expose core public symbols at package level."""
    import sigmutsel

    assert hasattr(sigmutsel, "MutationDataset")
    assert hasattr(sigmutsel, "Model")
    assert hasattr(sigmutsel, "locations")


def test_can_import_core_modules():
    """Import core modules without side effects raising."""
    from sigmutsel import locations  # noqa: F401
    from sigmutsel import models  # noqa: F401
