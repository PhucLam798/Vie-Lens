# E8 selection rule

A concept is competent when its unsteered greedy answer is scored VI under A10. Competent concepts are ranked within stratum by the frozen E8 order. Set n = min(e8_max_items, number of competent concepts). Allocate n proportionally to the competent stratum counts using largest remainder, breaking remainder ties by stratum name. Within each stratum, select the first quota competent concepts in rank order.

- `e8_max_items`: 300
- `seed`: 20260925
- E8 stream key: `e8|<family>|<stratum>`
- E0 sample stream key: `e0_sample`
- Main distractor key: `distractor|<family>|<lang>|<variant>|<concept_id>`
- E0 distractor key: `e0_distractor|<family>|<lang>|<variant>|<concept_id>`
