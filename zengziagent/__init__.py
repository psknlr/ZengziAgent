"""ZengziAgent: schema-conditioned multi-stage LLM framework for fine-grained peer-review annotation.

Pipeline (Algorithm 1 in the manuscript)::

    S_D  = Planner(G, E, metadata)            # structured task specification (Stage 1)
    P    = Actor(S_D, M) = R_M(C(S_D))         # prompt compilation + backend rendering (Stage 2)
    x'   = Preprocessor(x)                      # deterministic, offset-preserving normalisation (Stage 3)
    A    = Analyzer(P, x')                      # schema-constrained XML annotation (Stage 4)
    V    = Recorder.validate(A, S_D)            # validation + bounded refinement loop (Stage 5)
    A*   = TextAlignment(A, x)                  # Algorithm 2: source-span recovery + token mapping

Every stage emits an auditable artifact that is written to disk by the experiment runners.
"""

__version__ = "2.0.0"

from .schema import (  # noqa: F401
    LABELS,
    AnnotationSchema,
    AnnotationUnit,
    GoldSpan,
    PipelineConfig,
    ReviewRecord,
    TaskSpecification,
)
