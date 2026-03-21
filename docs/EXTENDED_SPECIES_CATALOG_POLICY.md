# Extended Species Catalog Policy

Product decision: the extended species catalog should use **eBird/Clements** as
the canonical taxonomy source. Scientific keys, species inclusion, and taxonomy
versioning are intended to come from a versioned eBird/Clements snapshot rather
than from iNaturalist.

Product decision: the long-term localized-name model is **international and
locale-keyed**. The catalog should not keep growing by adding more flat
`common_*` fields for each new language.

Until that migration is implemented, the shipped catalog remains a
maintainer-generated committed snapshot. The current generator still uses
iNaturalist taxa responses to build the asset, but that is transitional and
does not change the intended taxonomy policy.

**iNaturalist** is used only to enrich display names. It may supply localized
common names, but it does not define taxonomy or species identity.

Current transitional state:
- the shipped asset still uses flat fields such as `common_de`, `common_en`,
  and `common_nb`
- the current `common_nb` field is populated from iNaturalist `locale=nb` and
  therefore represents **Bokmal-oriented Norwegian names**

Target state:
- generated entries should use a locale-keyed name object, for example
  `names.en`, `names.de`, `names.nb`, `names.fr`, `names.es`

The application ships a **committed generated JSON catalog**. End users must
never generate or refresh taxonomy data locally; catalog refreshes are
**maintainer-only** and must be committed into the repository so every release
uses one consistent snapshot.

Runtime name fallback should be universal:
- `requested locale -> en -> scientific`

If app settings continue to expose `NO`, that should be treated as an app-level
alias that resolves to `nb` in the catalog name layer.

Every catalog refresh must record:
- the eBird/Clements taxonomy version
- the iNaturalist name-enrichment fetch date
- the total entry count
- missing-name counts per language
- notable taxonomy/key diffs

The total entry count is a **snapshot metric**, not a product promise.

Repository documentation must include:
- source provenance for taxonomy and localized names
- attribution/licensing notes for iNaturalist-based name enrichment
- the refresh procedure
- the validation checks required before committing a new catalog snapshot

## Current Snapshot Metadata

- Generated: `2026-03-21`
- Output asset: `assets/extended_species_global.json`
- Current generator locales:
  - `de -> common_de`
  - `nb -> common_nb`
- Current generator endpoint:
  - `https://api.inaturalist.org/v1/taxa?taxon_id=3&rank=species`
- Current snapshot counts:
  - total entries: `11360`
  - missing `common_de`: `1482`
  - missing `common_en`: `14`
  - missing `common_nb`: `714`
