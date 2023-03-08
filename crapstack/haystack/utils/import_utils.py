import logging
import importlib
from pathlib import Path


logger = logging.getLogger(__name__)


def safe_import(import_path: str, classname: str, dep_group: str):
    """
    Method that allows the import of nodes that depend on missing dependencies.
    These nodes can be installed one by one with project.optional-dependencies
    (see pyproject.toml) but they need to be all imported in their respective
    package's __init__()

    Therefore, in case of an ImportError, the class to import is replaced by
    a hollow MissingDependency function, which will throw an error when
    inizialized.
    """
    try:
        module = importlib.import_module(import_path)
        classs = vars(module).get(classname)
        if classs is None:
            raise ImportError(f"Failed to import '{classname}' from '{import_path}'")
    except ImportError as ie:
        classs = _missing_dependency_stub_factory(classname, dep_group, ie)
    return classs


def _missing_dependency_stub_factory(classname: str, dep_group: str, import_error: Exception):
    """
    Create custom versions of MissingDependency using the given parameters.
    See `safe_import()`
    """

    class MissingDependency:
        def __init__(self, *args, **kwargs):
            _optional_component_not_installed(classname, dep_group, import_error)

        def __getattr__(self, *a, **k):
            return None

    return MissingDependency


def _optional_component_not_installed(component: str, dep_group: str, source_error: Exception):
    raise ImportError(
        f"Failed to import '{component}', "
        "which is an optional component in Haystack.\n"
        f"Run 'pip install 'farm-haystack[{dep_group}]'' "
        "to install the required dependencies and make this component available.\n"
        f"(Original error: {str(source_error)})"
    ) from source_error

