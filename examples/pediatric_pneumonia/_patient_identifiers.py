"""File-name conventions of the pediatric chest X-ray archive, parsed explicitly.

A grouped split is only as trustworthy as the identifiers it groups by, so this
module refuses to guess. Every supported file-name pattern is written down here
together with what its numbers are documented to mean, and anything that does
not match a listed pattern becomes its own singleton group flagged
``UNKNOWN`` - never a confirmed, unique patient.

Two rules follow from the protocol and are enforced by construction:

1. **Identifiers from different conventions are never comparable.** Group ids
   are namespaced by the convention that produced them, so ``IM-0222`` from the
   ``NORMAL`` sub-collection can never merge with ``NORMAL2-IM-0222``.
2. **An undocumented number is not a patient id.** Only the ``person<N>``
   prefix used by the pneumonia images is treated as a per-patient counter.
   The ``IM-<study>-<image>`` names group images of one acquisition series,
   which is a safe *over*-grouping (it can only keep related images together,
   never merge two distinct series), and is reported as such rather than as
   patient identity.
3. **Counters are only comparable inside the collection that issued them.**
   The shipped train and test directories number their patients
   independently, so ``person1`` in one is not the person behind ``person1``
   in the other. Every key therefore carries an explicit
   ``identifier_scope``; two records can only land in the same group when
   their scopes agree. A consequence worth stating plainly in the report: with
   per-pool scopes, file names cannot confirm *or* rule out that a patient
   appears in both the development pool and the official test set.

References:
    Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying Medical
        Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*,
        172(5), 1122-1131. DOI: 10.1016/j.cell.2018.02.010
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class GroupConfidence(StrEnum):
    """How much the grouping key is actually known to mean.

    Attributes:
        CONFIRMED_PATIENT: The archive documents this number as a per-patient
            counter. Images sharing it are the same person.
        STUDY_SERIES: The number identifies one acquisition series. Images
            sharing it belong together, but the series is not documented to map
            one-to-one onto a patient, so patient independence is *not* proven
            by it.
        UNKNOWN: No supported pattern matched. The image becomes a singleton
            group and the audit reports it, so a run can never claim patient
            disjointness it did not establish.
    """

    CONFIRMED_PATIENT = "confirmed_patient"
    STUDY_SERIES = "study_series"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GroupAssignment:
    """The grouping key derived from one file name.

    Attributes:
        group_id: Namespaced key. Two records may only share a split-blocking
            group when this string is equal.
        confidence: What the key is known to mean; see :class:`GroupConfidence`.
        convention: Name of the matched pattern, recorded in the manifest so a
            reader can check the parse rather than trust it.
    """

    group_id: str
    confidence: GroupConfidence
    convention: str


# person<N>_<bacteria|virus>_<M>.jpeg - the pneumonia images. ``N`` is the
# per-patient counter of the source collection; ``M`` numbers that patient's
# images. This is the only pattern whose number is documented as patient-level.
_PERSON_PATTERN = re.compile(
    r"^person(?P<patient>\d+)_(?P<pathogen>bacteria|virus)_(?P<image>\d+)$",
    re.IGNORECASE,
)

# NORMAL2-IM-<study>-<image>[-<extra>].jpeg and IM-<study>-<image>[-<extra>].jpeg
# - the normal images, kept as two separate sub-collections. ``study`` indexes
# an acquisition series, not a person, so these parse to STUDY_SERIES.
_IMAGE_SERIES_PATTERN = re.compile(
    r"^(?P<collection>NORMAL\d*-IM|IM)-(?P<study>\d+)-(?P<image>\d+)(?P<extra>-\d+)?$",
    re.IGNORECASE,
)

# BACTERIA-<study>-<image>.jpeg / VIRUS-<study>-<image>.jpeg - present in some
# re-releases of the archive instead of the ``person`` naming. The long number
# is a study accession, so it is also STUDY_SERIES rather than patient-level.
_PATHOGEN_SERIES_PATTERN = re.compile(
    r"^(?P<collection>BACTERIA|VIRUS)-(?P<study>\d+)-(?P<image>\d+)$",
    re.IGNORECASE,
)

# Scope used when a caller states that the archive numbers its patients once,
# across every shipped directory. The pediatric pneumonia release does not, so
# the audit passes a per-pool scope instead.
GLOBAL_IDENTIFIER_SCOPE = "global"

SUPPORTED_CONVENTIONS: tuple[str, ...] = (
    "person_patient",
    "image_series",
    "pathogen_series",
)


def parse_group_assignment(
    file_stem: str,
    *,
    fallback_key: str,
    identifier_scope: str = GLOBAL_IDENTIFIER_SCOPE,
) -> GroupAssignment:
    """Derive the grouping key for one image from its file name.

    Args:
        file_stem: File name without its extension, e.g. ``person1_bacteria_1``.
        fallback_key: Stable, unique string (the record's relative path) used to
            build a singleton group when no convention matches. Passing the
            example id keeps unmatched images separable instead of silently
            collapsing them into one bucket.
        identifier_scope: Name of the collection whose counter issued the
            number. Keys from different scopes never merge. Pass
            :data:`GLOBAL_IDENTIFIER_SCOPE` only when the numbering really is
            documented as global across the whole archive.

    Returns:
        The namespaced :class:`GroupAssignment`. Callers must treat anything
        other than :attr:`GroupConfidence.CONFIRMED_PATIENT` as "patient
        independence not established" when writing up results.
    """
    normalized_stem = file_stem.strip()
    scope_prefix = f"{identifier_scope}|"

    person_match = _PERSON_PATTERN.match(normalized_stem)
    if person_match is not None:
        patient_number = int(person_match.group("patient"))
        return GroupAssignment(
            group_id=f"{scope_prefix}person:{patient_number}",
            confidence=GroupConfidence.CONFIRMED_PATIENT,
            convention="person_patient",
        )

    series_match = _IMAGE_SERIES_PATTERN.match(normalized_stem)
    if series_match is not None:
        collection = series_match.group("collection").upper()
        study_number = int(series_match.group("study"))
        return GroupAssignment(
            group_id=f"{scope_prefix}series:{collection}:{study_number}",
            confidence=GroupConfidence.STUDY_SERIES,
            convention="image_series",
        )

    pathogen_match = _PATHOGEN_SERIES_PATTERN.match(normalized_stem)
    if pathogen_match is not None:
        collection = pathogen_match.group("collection").upper()
        study_number = int(pathogen_match.group("study"))
        return GroupAssignment(
            group_id=f"{scope_prefix}series:{collection}:{study_number}",
            confidence=GroupConfidence.STUDY_SERIES,
            convention="pathogen_series",
        )

    return GroupAssignment(
        group_id=f"{scope_prefix}unknown:{fallback_key}",
        confidence=GroupConfidence.UNKNOWN,
        convention="unmatched",
    )
