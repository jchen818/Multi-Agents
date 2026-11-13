// Pass = the expected category is ranked in the top 3 of verified hypotheses. Score 1 for top-1, 0.5 for top-3.
module.exports.check = (output) => {
  let o;
  try { o = typeof output === 'string' ? JSON.parse(output) : output; } catch (e) { return { pass: false, score: 0, reason: 'bad output' }; }
  const got = (o.ranked || []).map(r => r.category).slice(0, 3).join(' > ') || 'nothing verified';
  if (o.top1) return { pass: true, score: 1, reason: `top-1 ${o.expected} (${o.calls} calls)` };
  if (o.top3) return { pass: true, score: 0.5, reason: `rank ${o.rank}: ${got} (expected ${o.expected})` };
  return { pass: false, score: 0, reason: `expected ${o.expected}, got ${got}` };
};
