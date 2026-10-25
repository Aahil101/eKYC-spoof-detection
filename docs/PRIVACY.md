# Privacy and compliance notes

This is a **research prototype**, not a certified KYC system and not legal advice. Before any real deployment, a lawyer and a DPIA are required.

## What is collected and where it goes

| Data | Stored as | Retention | Notes |
|---|---|---|---|
| ID document image/PDF, selfie video (with audio) | AES-256-GCM encrypted files in `MEDIA_DIR`, bound to session id + name (AAD) | `MEDIA_TTL_DAYS` (default 30) after processing; 1 day if never submitted | Key from `MEDIA_KEYS` or auto-generated `data/secrets/media.keys` (demo only). Key id in the file header allows rotation. |
| Quality vector, module features, scores, decision | SQL (`quality`, `module_outputs`, `sessions`) | `LOG_TTL_DAYS` (default 365) | Needed for drift monitoring. No raw pixels. |
| Artifact embeddings (CLIP / forensic / texture) | SQL `embeddings` | as logs | Used for novelty detection. Frozen CLIP features may still carry identity information; this is treated as personal data. |
| Face-match template (ArcFace) | Encrypted, separate table `identity_templates` | purged with media | Never used for monitoring or novelty. |
| OCR text from the ID | **not stored**; only validity flags (e.g. MRZ check-digit pass/fail) | — | Milestone 3. |
| Session token | SHA-256 hash only | as logs | |

Application logs are structured JSON. A scrubber redacts keys such as `embedding`, `image`, `name` and `id_number`, and no media is ever logged.

## User rights

- **Consent** is explicit and versioned (`CONSENT_VERSION`). The API rejects sessions without consent or with a stale consent version.
- **Erasure:** `DELETE /api/sessions/{id}` (the "Delete my data" button) removes media, templates, features, scores and labels.
- **Purpose limitation:** data is used for the KYC decision and for monitoring model quality only.

## GDPR (if EU data subjects are involved)

- Facial images processed for unique identification are special-category biometric data (Art. 9). The lawful basis assumed here is explicit consent (Art. 9(2)(a)).
- A DPIA (Art. 35) is required for large-scale biometric processing.
- Erasure (Art. 17) and storage limitation (Art. 5(1)(e)) are implemented as above.
- Automated decisions with legal effects (Art. 22): the policy routes uncertain cases to HUMAN_REVIEW. A production system must offer human intervention for every REJECT.

## India DPDP Act 2023 (Act 22 of 2023)

- Consent must be free, specific, informed and unambiguous, with an itemised notice. The consent screen lists what is processed and why.
- The data principal can withdraw consent and request erasure. The delete endpoint supports this.
- Data must be erased once the purpose is served, unless retention is required by law. **RBI V-CIP rules may require keeping recordings for longer.** `MEDIA_TTL_DAYS` is configurable for that reason, and the right value is a legal decision, not a technical one.
- Reasonable security safeguards: encryption at rest, token-scoped access, an admin API key outside demo mode, and localhost-only demo mode.

## Datasets and models

- Never use real people's ID documents. Use synthetic or specimen IDs (SIDTD, MIDV-2020-derived) and your own face only.
- Public datasets are used under their original licenses (see `docs/RESEARCH_NOTES.md`). Kaggle mirrors do not change those terms.
- InsightFace pretrained models are licensed for **non-commercial research only**. facenet-pytorch is a drop-in alternative behind the same M5 interface.

## Known gaps (milestone 2)

- No TLS inside the stack. Put a TLS-terminating proxy in front of any non-local deployment.
- The demo key file sits on the same disk as the data. In production, use a KMS or secret store.
- There is no audit log of admin access yet.
