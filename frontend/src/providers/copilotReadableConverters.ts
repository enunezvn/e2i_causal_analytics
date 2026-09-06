/**
 * Readable value converters for useCopilotReadable.
 *
 * CopilotKit react-core 1.51.2 types `convert(description, value)` but its
 * runtime calls `convert(value)` with ONE argument - dist/index.js and the
 * ESM chunk behind dist/index.mjs both read
 * `(convert != null ? convert : JSON.stringify)(value)` - so a positional
 * `(description, value) => String(value)` converter stringifies `undefined`.
 * Taking the LAST argument is the value under both signatures.
 */
export const passThroughText = (...args: unknown[]): string => String(args[args.length - 1]);
