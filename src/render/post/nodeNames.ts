/**
 * @module render/post/nodeNames
 *
 * One function, in its own module, because both `PostStack` and `Tonemap` need
 * it and they already import each other.
 *
 * ## Why it exists at all
 *
 * three emits a named uniform node's `name` **verbatim as the shader
 * identifier**: `UniformNode.generate` calls
 * `builder.getUniformFromNode(node, type, stage, this.name)`, and the returned
 * property name is written straight into the generated GLSL or WGSL. This
 * project's convention for naming things is dotted — `post.copy.source`,
 * `composite.bloom`, `lightShafts.march` — and a dot is not legal in a GLSL
 * identifier, so a texture node named that way emits
 *
 * ```glsl
 * uniform sampler2D post.copy.source;   // ERROR: 0:47: '.' : syntax error
 * ```
 *
 * The fragment shader then fails to compile, the renderer runs on an invalid
 * program (`INVALID_OPERATION: useProgram: program not valid`), and every draw
 * using that material writes nothing. A black frame, from a naming convention.
 *
 * That had already happened to `PostStack`'s copy material — the fallback used
 * when the stack is disabled, and when the active chain contains no chain pass.
 * Neither is reachable in a shipping configuration, so it sat there compiling
 * to black until `?post=off` made the first path reachable on purpose. Worth
 * remembering the next time a comment claims a fallback is safe: a guard that
 * has never executed is a guess.
 */

/**
 * Coerce a human-readable node name into a legal shader identifier.
 *
 * Non-alphanumerics become underscores and a leading digit is prefixed, so
 * `post.copy.source` becomes `post_copy_source` and the debugging value of the
 * name survives intact.
 */
export function safeNodeName(name: string): string {
  const cleaned = name.replace(/[^A-Za-z0-9_]/g, '_');
  return /^[A-Za-z_]/.test(cleaned) ? cleaned : `_${cleaned}`;
}
