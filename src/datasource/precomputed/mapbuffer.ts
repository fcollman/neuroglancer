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
 * @file Parser for the MapBuffer format (https://github.com/seung-lab/mapbuffer),
 * used by precomputed `.frags` files: a single indexed blob holding many
 * objects keyed by uint64 label. For spatially-indexed skeletons each `.frags`
 * file packs every skeleton fragment in one spatial chunk.
 *
 * On-disk layout:
 *   HEADER (16 bytes):
 *     [0,7)   magic  = "mapbufr"
 *     [7]     format version (0 or 1)
 *     [8,12)  compression: ascii "none" | "gzip" | "00br" | "zstd" | "lzma"
 *     [12,16) uint32le  number of keys
 *   INDEX: numKeys × (uint64le label, uint64le offset)
 *   DATA:  per-key blobs; offsets are absolute file byte positions, so blob i
 *          spans [offset_i, offset_{i+1}) (last → end of file). For format
 *          version 1 each blob is followed by a 4-byte crc32c that is included
 *          in the offset span and must be stripped.
 *
 * This module only parses structure; blob payloads remain compressed (per the
 * `compression` field) for the caller to decode.
 */

const MAGIC = "mapbufr";
const HEADER_SIZE = 16;
const INDEX_ENTRY_SIZE = 16; // uint64 label + uint64 offset
const CRC32C_SIZE = 4;

export type MapBufferCompression =
  | "none"
  | "gzip"
  | "00br" // brotli
  | "zstd"
  | "lzma";

export interface MapBufferEntry {
  label: bigint;
  /** Still-compressed blob bytes (crc32c already stripped for version 1). */
  data: Uint8Array<ArrayBuffer>;
}

export interface ParsedMapBuffer {
  version: number;
  compression: MapBufferCompression;
  entries: MapBufferEntry[];
}

function readMagic(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < MAGIC.length; ++i) s += String.fromCharCode(bytes[i]);
  return s;
}

function readCompression(bytes: Uint8Array): string {
  let s = "";
  for (let i = 8; i < 12; ++i) {
    const c = bytes[i];
    if (c !== 0) s += String.fromCharCode(c);
  }
  return s;
}

/**
 * Parses a MapBuffer file into its entries. Throws if the magic is invalid.
 */
export function parseMapBuffer(buffer: ArrayBuffer): ParsedMapBuffer {
  const bytes = new Uint8Array(buffer);
  if (buffer.byteLength < HEADER_SIZE || readMagic(bytes) !== MAGIC) {
    throw new Error("Invalid MapBuffer: bad magic");
  }
  const dv = new DataView(buffer);
  const version = bytes[7];
  const compression = readCompression(bytes) as MapBufferCompression;
  const numKeys = dv.getUint32(12, /*littleEndian=*/ true);

  const dataStart = HEADER_SIZE + numKeys * INDEX_ENTRY_SIZE;
  if (buffer.byteLength < dataStart) {
    throw new Error("Invalid MapBuffer: truncated index");
  }

  // Read (label, offset) pairs, then sort by offset to bound each blob.
  const labels = new Array<bigint>(numKeys);
  const offsets = new Array<number>(numKeys);
  const order = new Array<number>(numKeys);
  for (let i = 0; i < numKeys; ++i) {
    const base = HEADER_SIZE + i * INDEX_ENTRY_SIZE;
    labels[i] = dv.getBigUint64(base, /*littleEndian=*/ true);
    offsets[i] = Number(dv.getBigUint64(base + 8, /*littleEndian=*/ true));
    order[i] = i;
  }
  order.sort((a, b) => offsets[a] - offsets[b]);

  const crcSize = version >= 1 ? CRC32C_SIZE : 0;
  const entries: MapBufferEntry[] = new Array(numKeys);
  for (let k = 0; k < numKeys; ++k) {
    const i = order[k];
    const start = offsets[i];
    const end = k + 1 < numKeys ? offsets[order[k + 1]] : buffer.byteLength;
    const length = Math.max(0, end - start - crcSize);
    entries[k] = {
      label: labels[i],
      data: new Uint8Array(buffer, start, length) as Uint8Array<ArrayBuffer>,
    };
  }
  return { version, compression, entries };
}
