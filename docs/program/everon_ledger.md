# Everon exposure ledger

Pre-registration of every exposure to Everon, the sealed zero-shot test map, and of every decision that Everon could
have motivated (CAPTURE_PROGRAM r2 L4, RT-C18). An entry records what was seen or decided and the effect it may
have. Nothing Everon-derived may change training, selection or evaluation unless an entry here allows it first.

Seed: CAPTURE_PROGRAM r2.2 Appendix A, "Everon ledger seed (pre-registration, 2026-10-20) [RT-C18, L4]", rows EL-001
to EL-007, copied verbatim. Section references (&sect;) are to CAPTURE_PROGRAM r2; the source's section sign is
written as the HTML entity `&sect;`, so this file stays ASCII and renders as the source does. No Everon result exists
as of 2026-10-20.

Rules:

- **Append-only.** A row is never edited, reordered or removed. A correction is a new row that names the row it
  corrects.
- A new row goes at the end of the table, with the next id (EL-008, EL-009, ...), its date (never earlier than the
  row above it), the exposure or decision, and the effect allowed.
- A lever added after EL-007 needs a non-Everon justification logged here first (EL-007). An expert change after an
  Everon harness run is logged here once the failure is reproduced on Arland or Kolguyev, and a declared build
  migration (1.8 to 1.9, a 1.9.x hotfix) with the non-Everon justification "build change" (CAPTURE_PROGRAM r2.2 L11).
- Before a change is committed, `python3 -I -B tools/harvester/check_ledger_append_only.py` checks it against the
  committed version (`--base-rev`, default HEAD). The seed rows are also checked against their fixed digest.

| Id | Date | Exposure or decision | Effect allowed |
|---|---|---|---|
| EL-001 | 2026-10-20 | [MG] decoded Everon world statistics: road classes (E01 12 m dashed, cobblestone, bridges), vegetation mix, grades, settlements, navmesh streaming | stratify reports only; no training change targets them |
| EL-002 | 2026-10-20 | The user's ArmaDrive and ArmaDriveV2 are Everon SubScenes with waypoints across Everon; `SCR_LocalRoadExtractor` was run on Everon ([MG] &sect;1.3). The repo navigation-code lineage was developed on Everon | **no constants or thresholds are ported** from those projects; re-derive on Arland |
| EL-003 | 2026-10-20 | The Claude red-team ran a connectivity check on decoded Everon geometry (no frames); no training recommendation was drawn from it | none |
| EL-004 | 2026-10-20 | Pretrained encoders may have seen public Reforger footage, mostly Everon (L9) | disclosure; Kolguyev gap comparison |
| EL-005 | 2026-10-20 | Navmesh-streaming setting equalised between maps because Everon streams the vehicle navmesh | infrastructure parity only |
| EL-006 | 2026-10-20 | Absolute-coordinate inputs banned (L3). The rule is generic, but its motivation cited Everon's coordinate range | generic rule |
| EL-007 | 2026-10-20 | Generic diversity levers pre-registered (&sect;2.5) and the unseen-class stratum pre-declared as an expected failure (&sect;1.6) | as listed; any lever added later needs a non-Everon justification logged here first |
