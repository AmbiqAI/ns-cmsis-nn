"""
Custom exceptions for CMSIS-NN Tools.
"""


class CMSISNNToolsError(Exception):
    """Base exception for all CMSIS-NN Tools errors."""
    pass


class RepoRootNotFoundError(CMSISNNToolsError):
    """Raised when repository root cannot be found."""
    pass


class ConfigurationError(CMSISNNToolsError):
    """Raised when configuration is invalid."""
    pass


class PathNotFoundError(CMSISNNToolsError):
    """Raised when a required path does not exist."""
    pass


class StepExecutionError(CMSISNNToolsError):
    """Raised when a pipeline step fails."""
    pass


class BuildError(CMSISNNToolsError):
    """Raised when a build operation fails."""
    pass


class FVPRunError(CMSISNNToolsError):
    """Raised when FVP execution fails."""
    pass


class GenerationError(CMSISNNToolsError):
    """Raised when model generation fails."""
    pass
