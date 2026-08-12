"""Shared discourse functions; ECHO is observed only, never a generation goal."""
DEFINITIONS = {
    'INITIATE':'Open a new claim, question or angle not dependent on the previous contribution.',
    'EXTEND':'Accept the previous point and add a new implication or concrete example.',
    'CONTEST':'Challenge one specific claim with a reason or counterexample, including a question containing a rival reading.',
    'CONCEDE':'Accept a specific counterargument and state how your own prior position changes.',
    'REFRAME':'Keep the prior claim intact but place the subject in a different explanatory frame.',
    'ANSWER':'Answer the open question substantively, including a clear position on an interpretive question.',
    'CLARIFY':'Request or supply a definition, distinction or correction of a misunderstanding.',
    'ECHO':'Repeat existing content or acknowledge socially without adding content.',
}
GENERATION_MOVES = tuple(m for m in DEFINITIONS if m != 'ECHO')
OBSERVED_MOVES = tuple(DEFINITIONS)
LEGACY_MAP = {'AGREE_AND_EXTEND':'EXTEND','BUILD_ON':'EXTEND','DISAGREE_SPECIFIC':'CONTEST',
              'CONCEDE_AND_REVISE':'CONCEDE','REDIRECT':'REFRAME'}
LEGACY_AMBIGUOUS = {'ASK_QUESTION','SHARE_ANECDOTE','REACT_EMOTIONALLY'}


def planned_move(label):
    """Do not infer an ambiguous old plan's discourse function from its name."""
    return label if label in GENERATION_MOVES else LEGACY_MAP.get(label)
