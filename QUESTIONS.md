# Questions to pick

Pick one question, tell Vuk which one, and post your results within one week. Write how you will run it before you run it: which change, which tier, and what result would mean yes or no. Every question has a useful answer whichever way it turns out.

**What `magree` is.** A small language model plus three lookup tables and MoE (mixture of experts):

- a **word table** after block 1, with one row per word;
- a **gated table** inside attention in the last block: each attention head looks at the 4 words it attends to most, picks one of 16 codes for each, and reads the row for that word and code;
- an **n-gram table** after the last block, with rows for the last 2 and 3 words, whose output starts at zero.

It recalls stored facts better than 9 published memory modules. Its problem: it learned "The capital of France is Paris", and it also pushes "Paris" for "The currency of France is". The **relation gap** measures that, and higher is better.

| # | Question | Start from |
|---|---|---|
| 1 | Does the gated table matter? Replace it with a plain table that has one row per word and no codes, and check whether fact recall falls and whether the wrong-question mistake gets smaller. | `Agree` in `methods.py` |
| 2 | What does the n-gram table do? Remove it and check whether the model gives the right answer to the wrong question more often. Also check whether starting its output at zero matters. | `mtok` (no n-gram table), `ZeroEngram` |
| 3 | How much fact recall comes from the word table? Remove it and see how much recall falls and whether the wrong-question mistake gets smaller. | `mgram` (no word table) |
| 4 | Does our gate matter? Replace it with Engram's original gate and check recall of facts asked in new words. | `magreed` (no cosine) |
| 5 | Does it matter that each head chooses its code? Give each head a fixed random code instead and compare recall. | `Agree.pair` |
| 6 | How many codes does each word need? Try 4 and 64 codes instead of 16. | `C` in `Agree` |
| 7 | Each head reads rows for the 4 words it attends to most. With only 1 word, how much recall do we lose and how much faster is it? | `A` in `Agree`, `debug/speed.py` |
| 8 | Is it better to add the table's row inside attention (to the values) instead of after attention? The paper Memory Attention (arXiv 2609.28399) found values better. | `mmomez` puts MoME in the values |
| 9 | Do the gains come from the tables or from MoE? Remove MoE, give its weights to the tables, and compare. | `moe=8` in `magree`, `ttrip` |
| 10 | Where should the word table sit? Move it from after block 1 to the middle or the last block. | `magree1`, `mtrip1` |
| 11 | Do the parts do the same jobs in a bigger model? Switch each table off in a trained model at tier 1h and compare with tier 5m, where this has already been done. | `debug/ablate.py` |
| 12 | Does `magree` still win when it gets the same compute time as Engram? It takes 1.4 times Engram's time per step, so give Engram more steps. | `--steps`, `debug/speed.py` |

## What each answer tells us

1. **Recall falls:** the codes are what make the gated table work. **Recall stays:** a plain table is enough, and the paper can drop the codes.
2. **Leak grows:** the n-gram table is what protects the relation gap. **No change:** it can be removed to save weights.
3. Either way, it says how much of `magree`'s recall is a simple per-word lookup.
4. **Recall falls:** our gate is a real contribution. **No change:** the gate is not needed.
5. **Recall falls:** the head's choice carries information. **No change:** the codes only add more rows.
6. The curve shows whether more rows per word keep helping or stop.
7. **Little loss:** a cheaper `magree` with 1 word per head. **Big loss:** reading several words is needed.
8. **Values better:** the design should change. **After better:** our choice is right, against the published finding.
9. **Gains stay:** the tables do the work. **Gains go:** MoE does the work, and the tables need MoE.
10. Either way, it says whether depth matters for a per-word table.
11. **Same jobs:** the 5m result holds at 1h and goes in the paper. **Different jobs:** the paper must report both sizes.
12. **Still wins:** the gain is not just more compute. **Loses:** the gain costs compute.

## Before you start

- Read [SETUP.md](SETUP.md), then run `python3 tables.py 5m` to see the existing scores with no download.
- Try your change at tier 5m on a CPU first, for example `python3 env.py --method magree --tier 5m --lr 0.01`. Then run tier 1h on a GPU.
- Run lr .01 and .02 for every model you compare.
- Post the changed model next to `magree` on the 6 main scores (web loss, stored facts, facts in new wording, name far from the answer, relation gap, usual phrase), with the command you ran.
- [RESULTS.md](RESULTS.md) and [EXPERIMENTS.md](EXPERIMENTS.md) list what has already run.
