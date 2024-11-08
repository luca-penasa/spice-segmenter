def test_import_package():
    """Verify we can import the main package"""
    import spice_segmenter


def test_has_version():
    """Check that the package has an accesible __version__"""
    import spice_segmenter

    version = spice_segmenter.__version__
