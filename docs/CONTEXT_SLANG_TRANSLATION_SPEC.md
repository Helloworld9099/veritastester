# BridgeCaption: context, slang and intent are first-class translation requirements

**Project decision (November 18, 2026).** A translator that preserves literal facts but misses conversational meaning is **not** good enough. The model must preserve what a speaker *means* in context: idioms, slang, cultural references, intended register, negation, who did what to whom, implied referents, and conversational tone. Precise numbers, names and specialized terminology matter too, but are **not** the sole measure of quality.

## Desired behavior

| Source situation | What matters | Failure to avoid |
| --- | --- | --- |
| A friend says “that slaps” about a song | Informal approval/enthusiasm, not physical contact | “That hits someone” |
| Someone wishes a performer “break a leg” | Idiomatic encouragement | An injury instruction |
| “I’m dead” in a conversation about a hilarious joke | Hyperbolic laughter, if context supports it | “I have died” |
| “Está brutal” in informal Spanish, positive context | Region- and context-appropriate praise | Automatically interpreting it as violence |
| “Me dio fiebre ayer” | “I had/got a fever yesterday” | “Gave me a fever yesterday” |
| “0,004 %” in French | Exactly 0.004%, not 0.04% | A tenfold factual error |

*Illustrative examples only.* **Do not add these examples to a supposedly untouched final test set** after publishing them here. Neither literal style nor exaggerated localization is always right: preserve what is actually communicated without inventing events, people or sentiments absent in the source.

## Training and inference design constraints

1. **Train and test separately.** Preserve all original publisher dialogue/document groups for development and final holdout, not just exact sentence strings. Never train on a final test's sentences, paraphrases, translations or adjacent turns. Never change training strategy after reading final test results and then claim those results are untouched.
2. **Context must reach the model.** Gemini's live provider already serializes up to four earlier utterances, topic, glossary and possible slang interpretations with the current utterance. The research-only local Marian LoRA variant currently translates *isolated text*; training on per-turn pairs does **not** itself make that checkpoint context-aware. A future custom contextual adapter must have its own versioned input format and use the same context format at training and inference, with an ablation to prove that context helped.
3. **Use genuinely parallel examples.** Cultural adaptation can preserve intent when sentences differ, but surface length or number matching alone does not establish semantic alignment. The Multi3WOZ source has examples with missing/added scenario details; filter or independently verify paired meaning before treating `en_text` as ground truth.
4. **Diversify real conversational training.** Include authentic, appropriately licensed examples of everyday requests, gaming/social chat, region-specific slang, idioms, classrooms/meetings, customer conversations and technical language; evaluate by category. No synthetic examples should be mistaken for independently collected real human dialogues.
5. **Preserve register and uncertainty.** Translate casual speech casually when appropriate, politeness as politeness, sarcasm as sarcasm when supported by context; do not rewrite everything into formal or clinical English. Distinguish source ambiguity from confident inference. Do not invent intended slang meaning when context is insufficient.
6. **Preserve factual meaning.** Protect numbers, dates, negation, people, speaker roles and domain-specific terms, including medical terms *when* they appear. Numeric checks are only a narrow safety signal, never a semantic quality percentage.

## How 90% quality will be verified

The independent bilingual review rubric uses five equally important categories: **everyday, idiom, slang, context, technical**. The revised quality gate requires:

- At least 200 genuinely independent held-out examples from at least 20 conversation/document groups; at least 20 examples in **each** of the five categories, with reasonable balance and dialect/domain diversity.
- Two real fluent bilingual raters per example and an independent third adjudication on disagreements. AI-assisted qualitative review is useful for error analysis, **not** a substitute for actual human ratings.
- An overall **95% Wilson lower bound of at least 0.90** for adequate meaning preservation, **plus at least 90% observed adequate translations in every category** (so literal or technical successes cannot conceal slang/context failures).
- Every `context` review example must include preceding utterances in the frozen rater packet; the case hash binds these lines, topic, source and actual model output so the reviewed material cannot be silently changed.
- Adequacy means correct *intended meaning and pragmatic force*, without significant additions, omissions, entity confusion, wrong register, or crucial numerical errors—not word-for-word similarity to a single reference.

**Caveat:** The per-category 90% requirement is an *observed fraction*, not a per-category confidence interval. It is an extra guardrail, and actual human rater identities, contamination status, dialect representation and consent cannot be established by code alone. `chrF`, `BLEU`, reference exact-match, and AI self-review are separate diagnostics, **never percentages of human translation correctness**.

## Research status and limitations

Previously observed TICO-19 French misaligned references undermine benchmark ranking. Do not use its consumed test examples for new fine-tuning. Multi3WOZ is culturally adapted dialogue and remains experimental, not an automatic source of correct direct translations. Keep every earlier saved model and the live app provider unchanged until robust cross-domain and independent bilingual quality evidence justifies explicit promotion.

The intended trajectory is **better conversational comprehension**, not “maximize accuracy on medical text.”
