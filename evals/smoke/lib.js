function calls(o) {
  if (Array.isArray(o)) return o;
  if (o && o.tool_calls) return o.tool_calls;
  if (typeof o === 'string') { try { const j = JSON.parse(o); return Array.isArray(j) ? j : (j.tool_calls || []); } catch (e) { return []; } }
  return [];
}
const name = c => (c.function ? c.function.name : c.name);
function args(c) { const a = c.function ? c.function.arguments : c.arguments; try { return typeof a === 'string' ? JSON.parse(a) : (a || {}); } catch (e) { return {}; } }
const seen = o => JSON.stringify(calls(o).map(c => [name(c), args(c)]));

module.exports.expectTool = (output, ctx) => {
  const v = ctx.vars;
  const want = String(v.expect_tool).split('|');
  const hit = calls(output).find(c => want.includes(name(c)) &&
                                      (!v.expect_endpoint || args(c).endpoint === v.expect_endpoint));
  return { pass: !!hit, score: hit ? 1 : 0, reason: hit ? `called ${name(hit)}` : `got ${seen(output)}` };
};
module.exports.noTool = (output) => {
  const cs = calls(output);
  return { pass: cs.length === 0, score: cs.length ? 0 : 1, reason: cs.length ? `unexpected ${seen(output)}` : 'no tool call' };
};
module.exports.noEndpoint = (output, ctx) => {
  const bad = calls(output).find(c => args(c).endpoint === ctx.vars.bad_endpoint);
  return { pass: !bad, score: bad ? 0 : 1, reason: bad ? `invented endpoint ${ctx.vars.bad_endpoint}` : 'ok' };
};

