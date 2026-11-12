// One checker for all planner cases. Each case sets only the vars it needs:
//   expect_status "A|B", expect_tier "a|b", expect_any "r1|r2" (a step from these), only "r1|r2" (every step from these),
//   prefix "hbm_|hsio" (every step starts with one), forbid "burnin|lvl5" (no step contains), max_steps N,
//   based_on "seed" | "none", no_approval true, approval_text "a|b" (in the LAST approval shown), report_text "a|b",
//   recipe_saved true|false
const any = s => String(s).split('|');

module.exports.check = (output, ctx) => {
  const v = ctx.vars, fails = [];
  let o;
  try { o = typeof output === 'string' ? JSON.parse(output) : output; } catch (e) { return { pass: false, score: 0, reason: 'bad output' }; }
  const steps = o.steps || [];
  const shown = (o.approvals || []).length > 0;

  // Always: an approval view never shows a recipe outside the allowlist.
  if (shown && !o.all_allowed) fails.push(`non-allowlisted recipe shown: ${steps}`);

  if (v.expect_status && !any(v.expect_status).includes(o.status)) fails.push(`status ${o.status}, expected ${v.expect_status}`);
  if (v.expect_tier && !any(v.expect_tier).includes(o.tier)) fails.push(`tier ${o.tier}, expected ${v.expect_tier}`);
  if (v.expect_any && !steps.some(s => any(v.expect_any).includes(s))) fails.push(`steps ${steps} lack one of ${v.expect_any}`);
  if (v.only && steps.some(s => !any(v.only).includes(s))) fails.push(`steps ${steps} not all in ${v.only}`);
  if (v.prefix && steps.some(s => !any(v.prefix).some(p => s.startsWith(p)))) fails.push(`steps ${steps} not all ${v.prefix}*`);
  if (v.forbid && steps.some(s => any(v.forbid).some(f => s.includes(f)))) fails.push(`forbidden step in ${steps}`);
  if (v.max_steps && steps.length > Number(v.max_steps)) fails.push(`${steps.length} steps > ${v.max_steps}`);
  if (v.based_on === 'seed' && o.based_on !== o.seed_id) fails.push(`based_on "${o.based_on}", expected seed ${o.seed_id}`);
  if (v.based_on === 'none' && o.based_on) fails.push(`based_on "${o.based_on}", expected none`);
  if (v.no_approval && shown) fails.push('an approval was shown');
  if (v.approval_text) {
    const last = (o.approvals || []).slice(-1)[0] || '';
    if (!any(v.approval_text).some(t => last.toLowerCase().includes(t.toLowerCase()))) fails.push(`last approval lacks ${v.approval_text}`);
  }
  if (v.report_text && !any(v.report_text).some(t => (o.report || '').toLowerCase().includes(t.toLowerCase())))
    fails.push(`report lacks ${v.report_text}`);
  if (v.recipe_saved === true && !o.recipe_saved) fails.push('no recipe saved');
  if (v.recipe_saved === false && o.recipe_saved) fails.push('recipe saved');
  return { pass: !fails.length, score: fails.length ? 0 : 1,
           reason: fails.join('; ') || `ok (${o.status}, ${o.tier}: ${steps.join(',') || '-'})` };
};
