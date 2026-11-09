STEP 3 -- rate severity, defaulting to ACCEPT. Only a CLEARLY major problem (many cells affected) that a parameter can plausibly fix warrants a retune; tolerate minor imperfections, a few missed/false masks, and any merely good-enough result with severity='minor' or 'none' and param='none'. A retune that might not help is not worth it.

STEP 4 -- if severity is 'major', use the TOOL CARD to choose the SINGLE parameter and direction whose documented effect best fixes the problem you named; otherwise param='none', direction='none'.
{card_section}
Report: specimen; cell_size; cell_density; image_usable; problem (one short phrase or 'none'); severity; param; direction; confidence; and a one-sentence reasoning citing what you see in the overlay.

Deterministic metrics -- evidence you MAY weigh (e.g. a cell_count wildly inconsistent with the cells you see corroborates over/under-segmentation); judge holistically from the overlay and do not nitpick fine mask shape: {metrics_json}