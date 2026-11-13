// One checker for all Lab Ops cases. Each case sets only the vars it needs:
//   expect_tool "a|b", expect_arg "x|y", forbid_tool "a|b", no_tools true,
//   bad_arg "x", max_calls N, expect_text "a|b" (any), expect_all "a|b" (all),
//   live_cmd "cmd" (+ live_re): the answer must contain the value read live from the node (versions are not pinned)
const any = s => String(s).split('|');

function parse(o) {
  try { return typeof o === 'string' ? JSON.parse(o) : o; }
  catch (e) { return { answer: String(o), calls: [] }; }
}

module.exports.check = (output, ctx) => {
  const v = ctx.vars, o = parse(output), calls = o.calls || [];
  const ans = (o.answer || '').toLowerCase(), fails = [];
  const has = (c, a) => JSON.stringify(c.args || {}).toLowerCase().includes(a.toLowerCase());

  if (v.expect_tool) {
    const names = any(v.expect_tool), args = v.expect_arg ? any(v.expect_arg) : null;
    if (!calls.some(c => names.includes(c.name) && (!args || args.some(a => has(c, a)))))
      fails.push(`expected ${v.expect_tool}${v.expect_arg ? ' with ' + v.expect_arg : ''}`);
  }
  if (v.forbid_tool) {
    const bad = calls.filter(c => any(v.forbid_tool).includes(c.name));
    if (bad.length) fails.push(`forbidden: ${bad.map(c => c.name)}`);
  }
  if (v.no_tools && calls.length) fails.push(`expected no tools, got ${calls.map(c => c.name)}`);
  if (v.bad_arg && calls.some(c => has(c, v.bad_arg))) fails.push(`tool called with "${v.bad_arg}"`);
  if (v.max_calls && calls.length > Number(v.max_calls)) fails.push(`${calls.length} calls > ${v.max_calls}`);
  if (v.expect_text && !any(v.expect_text).some(t => ans.includes(t.toLowerCase())))
    fails.push(`answer lacks one of: ${v.expect_text}`);
  if (v.live_cmd) {
    if (!o.expected) fails.push(`could not read the live value with "${v.live_cmd}"`);
    else if (!ans.includes(String(o.expected).toLowerCase())) fails.push(`answer lacks live value ${o.expected}`);
  }
  if (v.expect_all) {
    const miss = any(v.expect_all).filter(t => !ans.includes(t.toLowerCase()));
    if (miss.length) fails.push(`answer lacks: ${miss}`);
  }
  return {
    pass: !fails.length,
    score: fails.length ? 0 : 1,
    reason: fails.join('; ') || `ok (${calls.map(c => c.name).join(', ') || 'no tools'})`,
  };
};
