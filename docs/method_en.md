# Method Notes

Consistency-gated self-correction treats self-correction as an exception rather than a default action. The direct answer is kept unless sampled evidence suggests that the answer is not reproducible under stochastic decoding.

The gate is simple:

1. Generate one deterministic direct answer.
2. Generate K sampled answers from the same prompt.
3. Compute the fraction of sampled answers that match the direct answer.
4. Replace the direct answer with the sampled majority answer only if the fraction is below the threshold.

This avoids a common failure mode of standard self-correction: unconditionally asking the model to revise can turn correct answers into wrong answers.
