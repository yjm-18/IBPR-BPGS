# Result Interpretation

The tables cover FOR-instance, Ofental, and LAUTx with TL (TreeLearn) and SAT
(SegmentAnyTree). `baseline` denotes the fixed base segmentation, `A` denotes the
first-stage variant, and `AB` denotes the full-method variant. Algorithmic details
are withheld in this preview.

| Column | Meaning | Unit |
|---|---|---|
| G | Net corrected points relative to the corresponding fixed baseline | Points |
| NIR | Net correction divided by the number of changed evaluation labels | Percent |
| IoU | Mean best instance IoU over reference trees | Percent |
| mAP | Mean instance average precision over the evaluation IoU thresholds | Percent |

A blank CSV NIR or JSON null denotes an undefined ratio when no labels change; it
must not be interpreted as zero. Baseline G is zero by definition. All reported
percentages use the 0-100 scale. Source numerical precision is preserved in CSV/JSON;
Markdown tables are rounded for readability.

Baseline P/R/F1 are calculated after pooling scene detection counts, not by averaging
scene-level percentages. They describe the baselines used in these experiments and
are not presented as numbers copied from the original model papers.

Differences between percentages are percentage-point differences. A higher G or NIR
need not imply higher IoU or mAP. The preview preserves outcomes where AB ties or
underperforms A; it does not select only favourable comparisons. The Ofental A and
AB rows coincide in this snapshot. The mechanism behind that selection is not part
of the public preview.

The published data consist only of aggregate measurements and evaluation scale.
Per-scene outputs, per-tree statistics, point-level changes and model selection
histories are not included. Complete evaluation code and procedural details are
reserved for a later method release.
