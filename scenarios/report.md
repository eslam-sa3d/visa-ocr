# Scenario test report

Engine setting: `auto`  |  scenarios: 61  |  PASS 53  PARTIAL 8  FAIL 0  |  pass rate 87%

## Matrix (✓ all fields right, ~ read but a field is wrong, ✗ not read)

| Document type | clean | rot+4 | rot-3 | blur | noise |
|---|---|---|---|---|---|
| passport:TD3 | ✓ ~ ✓ ✓ ✓ | ✓ ~ ✓ ✓ ✓ | ✓ ✓ ✓ ✓ ✓ | ✓ ✓ ✓ ✓ ✓ | ✓ ✓ ✓ ✓ ✓ |
| visa:MRVA | ✓ ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ | ✓ ✓ |
| id_card:TD1 | ~ ✓ | ✓ ✓ | ~ ✓ | ✓ ✓ | ✓ ✓ |
| id_card:TD2 | ~ | ✓ | ~ | ~ | ✓ |
| visa:MRVB | ✓ | ✓ | ✓ | ✓ | ✓ |
| iqama:visual | ✓ ✓ | ✓ ~ | – | ✓ ✓ | – |

## Pass rate by variant (bad-photo condition)

| Variant | Pass | Total |
|---|---|---|
| clean | 10 | 13 |
| rot+4 | 11 | 13 |
| rot-3 | 9 | 11 |
| blur | 12 | 13 |
| noise | 11 | 11 |

## Failures and partial reads

- **passport_egy__clean** (PARTIAL): givenNames: expected 'OMAR TEST', got 'OMAR TESI'
- **passport_egy__rot+4** (PARTIAL): givenNames: expected 'OMAR TEST', got 'OMAR TESI'
- **greencard_usa__clean** (PARTIAL): documentNumber: expected 'SRC0000000001', got 'SRC000000000L'
- **greencard_usa__rot-3** (PARTIAL): documentNumber: expected 'SRC0000000001', got 'SRCOO00000001'; givenNames: expected 'RAVI TEST', got 'RAVI TEST SSSBS'
- **idcard_td2_egy__clean** (PARTIAL): givenNames: expected 'AHMED TEST', got 'AHMED TESI'
- **idcard_td2_egy__rot-3** (PARTIAL): documentNumber: expected 'A1234567', got '41234567'; surname: expected 'SPECIMEN', got 'PECIMEN'
- **idcard_td2_egy__blur** (PARTIAL): givenNames: expected 'AHMED TEST', got 'AHMED TESI'
- **iqama_2__rot+4** (PARTIAL): idNumber: expected '2456789011', got None; nameEn: expected 'MARIAM NOOR SAEED', got None; nameAr: expected 'مريم نور سعيد', got None

Average time per image: 5.1s (slowest 16.1s).
