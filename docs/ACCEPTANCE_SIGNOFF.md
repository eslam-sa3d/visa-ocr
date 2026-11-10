# Go-live acceptance: decision sheet (checklist E2)

For the business owner. Each line is a threshold the OCR service must meet on REAL documents before it is used for decisions.
Tick "Agree" or write a different number. Once signed, the values are copied into `config/acceptance.json` and the status line is changed.

| # | Threshold | Proposed | Plain meaning | Agree / change to |
|---|---|---|---|---|
| 1 | Fully correct per document family | 95% | Out of 100 passports (and the same for each visa, ID and Iqama family), at least 95 come out with every field right | |
| 2 | Wrong value accepted without a person checking | 0 | The service must never pass a wrong value straight through without review | |
| 3 | Error rate among documents accepted without review | at most 1% | If a sample of auto-accepted documents is checked by hand, at most 1 in 100 has a wrong field | |
| 4 | Iqama key fields (number, name, dates) | 90% | Lower than passports because Iqama has no machine-readable lines; every Iqama is still reviewed by a person | |
| 5 | Median time per document | 10 seconds or less | Measured on the production server, not a laptop | |
| 6 | Minimum real specimens per family | 30 | Fewer than 30 real documents in a family means the result is not accepted as proof | |
| 7 | Families the bar applies to | Passports (by country group), Schengen and UK visas, US green card, Egyptian ID, Iqama, Saudi ID, other GCC IDs | Families with fewer than 30 specimens go live in "review everything" mode only | |

## Decisions needed
1. Is 95% right for passports, or does the business want higher (for example 98%)? A higher bar needs more real data and fine-tuning.
2. Who samples auto-accepted documents (rate 1 in N), and who owns the review queue?
3. May documents with a failed check digit ever be accepted? Proposed: never.
4. Until a family passes, do we run it in "review everything" mode? Proposed: yes.

## Caveats the owner should know
- Current numbers come from generated documents (about 91% fully correct, 0 wrong accepted) and a handful of real ones. They are not a claim of real-world accuracy.
- GCC ID layouts other than the Iqama are unverified: they always go to a person.
- The service does not replace a person for any document family until it passes this sheet.

Approved by: ____________________  Role: ______________  Date: ____________
