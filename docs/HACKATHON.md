# Veritas AI 2026 — submission and integrity checklist

Track: **AI + Languages**. Goal: Help people understand one another across language barriers, including accents and languages poorly supported by existing tools.

Rules reference: https://highschoolhackathon.org/

- All submitted project work must occur during the November 14–15, 2026 event. This repository was initialized and coded during that window.
- AI must perform the main job. Faster-Whisper recognizes audio and Gemini generates translations; the application is not a conventional site with an AI-themed wrapper.
- Pretrained models, public datasets, AI APIs, open-source libraries, and AI coding assistants may be used, but must be credited.
- Use public speech or recordings with consent. Do not record participants without permission or store confidential meetings.
- Report **actual** baseline vs improved measurements. Show examples next to correct references, break results out by language/accent, and identify failures.
- Explain technical limits and what was and was not trained.

## Provenance

The application uses pretrained speech and translation models. It does **not** fine-tune them. It includes a genuinely trainable tiny supervised conversation-topic classifier using original synthetic labelled examples. Its measured performance may be poor; report the result and do not present its score as translation accuracy. It serves only as an optional contextual topic hint.

The project used ChatGPT GPT-6 extensively for development, including generation of source files, tests, initial sample data and documentation during the hackathon window. Acknowledge this honestly in the submission and make sure team members can explain the code.

## Demonstration (around 2 minutes)

1. Explain the user: a student or worker watching a meeting in another language.
2. Launch the visible overlay; choose a microphone or system-audio monitor.
3. Play a licensed or permissioned foreign-language clip and show original/translated captions.
4. Test a slang or technical-terminology example with and without context or a glossary.
5. Show measured per-language WER and context/no-context translation results, including at least one failure.
6. Mention audio chunk latency, provider rate limits, model quality disparities, and privacy.

## Before submission

- [ ] Test on a real Linux workstation with microphone/system audio.
- [ ] Configure a real Gemini Developer API key and validate the chosen model ID.
- [ ] Confirm a usable subtitle appears over an actual meeting/video window.
- [ ] Collect genuine held-out evaluation results for multiple languages.
- [ ] Report errors and limitations; don't invent numbers.
- [ ] Record a demonstration, fill in the submission form, and make the code accessible to judges.
- [ ] Confirm project and submission comply with the event's final published rules.
