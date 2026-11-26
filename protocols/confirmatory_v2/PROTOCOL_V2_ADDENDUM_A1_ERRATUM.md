# Erratum to INTERPRETATION_REVIEW_V2.md and PROTOCOL_V2_ADDENDUM_A1.md (recorded before access to weeks 47-52)

**Wrong:** both documents say "12 view-restricted pair types".

**Correct:** analysis_v2.py (`VIEW_RESTRICTED_PAIRS`) defines **10** view-restricted pair types, 48 individual pairings in all:

| Pair types | Number of types | Individual pairings |
|---|---|---|
| Seed redundancy: MS+MS, MC+MC, LS+LS, LC+LC | 4 | 3 seed pairs each, 12 in total |
| Algorithm swap: MS+LS, MC+LC | 2 | 6 ordered seed combinations each, 12 in total |
| View swap: MS+MC, LS+LC | 2 | 6 ordered seed combinations each, 12 in total |
| Both factors swapped: MS+LC, LS+MC | 2 | 6 ordered seed combinations each, 12 in total |

**Effect:** none on any computation, hypothesis or threshold. IH3 (primary) uses the 10 type means. S-C3 uses the 48 individual pairings.
