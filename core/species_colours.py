"""Species colour slot assignment.

Deterministically maps scientific names to palette slots 0..7 so the
same species always gets the same Wong (2011) colour token everywhere in
the app — Review, Gallery, Stream, Trash, Subgallery.

This helper was originally inline in ``web/blueprints/review.py``. It is
extracted here so Gallery / Stream / Trash routes can import it without
creating a circular dependency on the Review blueprint.
"""

#: Number of distinct Wong palette slots.  Must match the number of
#: ``--species-colour-N`` CSS custom properties defined in
#: ``assets/design-system.css`` and the ``SPECIES_COLOURS`` array in
#: ``assets/js/gallery_utils.js``.
SPECIES_COLOUR_SLOTS: int = 8


def assign_species_colours(species_keys: "list[str] | set[str]") -> "dict[str, int]":
    """Deterministically map scientific names to palette slots 0..7.

    Contract:
    - Same input set → same output map (stable across reloads).
    - Alphabetical sort determines assignment order.
    - Wraps at :data:`SPECIES_COLOUR_SLOTS` when > 8 distinct species.
    - Empty / falsy species keys are ignored.
    - An empty input returns an empty map.

    Args:
        species_keys: Iterable of scientific-name strings.

    Returns:
        ``dict[species_key, slot_index]`` where ``slot_index`` is in
        ``range(SPECIES_COLOUR_SLOTS)``.
    """
    unique = sorted(
        {
            str(key).strip()
            for key in (species_keys or [])
            if key and str(key).strip()
        }
    )
    return {
        species_key: index % SPECIES_COLOUR_SLOTS
        for index, species_key in enumerate(unique)
    }
