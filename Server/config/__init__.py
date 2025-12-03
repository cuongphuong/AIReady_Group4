"""
Configuration package for bug classification
"""
from .bug_labels import BUG_LABELS, TEAM_GROUPS, LABEL_TO_TEAM
from .examples import FEW_SHOT_EXAMPLES

__all__ = ['BUG_LABELS', 'TEAM_GROUPS', 'LABEL_TO_TEAM', 'FEW_SHOT_EXAMPLES']
