/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Brotli decompression. The browser's `DecompressionStream` does not
 * support brotli, and neuroglancer's other codecs (`fflate`, `fzstd`,
 * `numcodecs`) don't cover it either, so we use the pure-JavaScript `brotli`
 * decoder. Pure JS (rather than a wasm decoder) keeps it working uniformly in
 * the browser and under `@vitest/web-worker` (where wasm codecs are known to
 * fail — see `decode_zstd_node.ts`).
 *
 * Used to decode the per-blob payloads of precomputed `.frags` MapBuffer files,
 * which flywire stores with `00br` (brotli) compression.
 */

import brotliDecompress from "brotli/decompress.js";

export function decodeBrotli(data: Uint8Array): Uint8Array<ArrayBuffer> {
  // The pure-JS decoder returns a Uint8Array backed by its own ArrayBuffer.
  return brotliDecompress(data) as Uint8Array<ArrayBuffer>;
}
