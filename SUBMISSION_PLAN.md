# Submission plan

Checked 25 Sep 2026. **Confirmed** dates come from the venue's own site;
*estimated* dates follow the venue's usual pattern and must be rechecked when
its call for papers appears.

Rule for the whole plan: the paper may be under review at **only one archival
venue at a time** (SPL, EUSIPCO, LoG proceedings). Non-archival workshops can
run in parallel only if both venues' policies allow it.

## Plan A: IEEE Signal Processing Letters (now)

| Date | Step |
|---|---|
| by **Fri 2 Oct 2026** | Make the GitHub repo public; submit to SPL (EDICS in README); complete the copyright form |
| optional, same week | Post the preprint to arXiv (needs an endorser; IEEE allows preprints) |
| ~9–23 Oct 2026 *(est.)* | Desk-rejection window. Silence after this means it went to reviewers |
| ~6–20 Nov 2026 *(est.)* | First decision (reported average 1.3 months) |
| within the deadline in the letter | Revise and resubmit; second decision ~4–6 weeks later |
| ~Jan–Feb 2027 *(est.)* | Final decision (reported average 2.7 months to acceptance) |

If accepted, you can also present it: EUSIPCO 2027 takes published journal
papers for presentation (**deadline 30 Apr 2027**, confirmed), and SPS
conferences usually offer an SPL presentation track (check ICASSP/ICIP 2027).

**If SPL rejects at any point → Plan B.** Rejections before ~20 Jan 2027 leave
enough time for EUSIPCO.

## Plan B: EUSIPCO 2027 (first fallback)

European Signal Processing Conference, Darmstadt, Germany,
**30 Aug – 3 Sep 2027** (confirmed). Archival (IEEE Xplore), about 5 pages,
closest in format to SPL.

| Date | Step |
|---|---|
| within 1 week of the SPL rejection | Read the reviews; list every fixable point |
| by ~20 Jan 2027 | Revise: address the SPL reviews, move to the EUSIPCO template, re-run `make_numbers.py`, recompile |
| ~late Jan 2027 | Check the call for papers (review model, page limit, AI policy, dual-submission rules) |
| **Sat 6 Feb 2027** | **Paper deadline** (confirmed) |
| ~mid–late May 2027 *(est.; 2026 was 19 May)* | Decision |
| 2 Jun 2027 | Author registration / early bird (confirmed); someone must register and present |

Costs: conference registration plus travel to Germany (or check for a remote
presentation option). Budget for this before submitting.

**Optional in parallel: ICLR 2027 workshop.** Double-blind, usually
non-archival. Workshops run **29–30 Apr 2027**, and every workshop must notify
authors by **26 Feb 2027** (both confirmed). Individual deadlines are usually
late January to early February 2027. Pick one on sequence models or graph
learning, and check that it allows papers under review elsewhere.

**If EUSIPCO rejects (~late May 2027) → Plan C.**

## Plan C: Learning on Graphs 2027 (second fallback)

Double-blind (OpenReview). Proceedings in PMLR: up to 9 pages plus unlimited
references and appendix. There is also a non-archival 4-page extended-abstract
track. In 2022, 29% of proceedings papers and 38% of extended abstracts were
accepted.

| Date | Step |
|---|---|
| Jun–Jul 2027 | Expand to 9 pages: full proofs, related work, experiment details, the per-lag tables now in `results/` |
| ~early Aug 2027 *(est.; 2026 was 5 Aug abstract / 7 Aug paper)* | Abstract and paper deadlines |
| ~mid Sep 2027 *(est.)* | Decision |
| ~late Nov 2027 *(est.)* | Conference; proceedings authors must attend in person |

If attending in person is not possible, use the extended-abstract track
(non-archival) and keep the paper free for a journal.

## Plan D (only if C also fails)

- **IEEE Open Journal of Signal Processing:** rolling submissions,
  single-blind. It charges an **APC of $2,160** (2026), with a 20% discount
  for IEEE society members.
- **AISTATS 2028:** double-blind, 8 pages, about 25% acceptance. The deadline
  is usually early October (2027 was 8 Oct 2026).

## Timeline at a glance

```
2026  Oct 2    submit SPL
      Oct 9-23 desk-reject window
      Nov      SPL first decision
2027  Jan-Feb  SPL final decision         --> if rejected: Plan B
      Feb 6    EUSIPCO 2027 deadline      (+ optional ICLR workshop)
      Feb 26   ICLR workshop decisions
      Apr 30   EUSIPCO journal-presentation deadline (if SPL accepted)
      May      EUSIPCO decision           --> if rejected: Plan C
      Aug      LoG 2027 deadline (est.)
      Aug 30   EUSIPCO 2027 conference
      Sep      LoG decision (est.)        --> if rejected: Plan D
      Nov      LoG 2027 conference (est.)
```
