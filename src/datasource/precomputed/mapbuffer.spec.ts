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

import { describe, expect, it } from "vitest";
import { parseMapBuffer } from "#src/datasource/precomputed/mapbuffer.js";
import { decodeBrotli } from "#src/util/brotli.js";

// A real per-blob payload (brotli-compressed precomputed skeleton, including
// the trailing 4-byte crc32c) taken verbatim from the first entry of
// gs://flywire_v141_m783/skeletons_mip_1/17910-18422_8912-9424_3088-3600.frags.
// 273 bytes; decompresses to a 616-byte skeleton (22 vertices, 21 edges).
const REAL_BLOB_BASE64 =
  "G2cCAMSKsbavqPlh2sgiqqFAJTLfMYcSTz44oAKNLMpvhZBNpBbQIQ9HNqhTc4wWwqullywAmc8Y" +
  "EfWHabMfvv59MSbqD9VmP/DBL6auNg3RkonWfBOw3TyYakfqDHO4cusCp3WJFT1Zg1hkgwB63aKy" +
  "7tQ9DgghtR7NE+aZdgNgIW22w4DTTGvmLHpzq3zyK6CgQgoroqhiiiuhpFJK58/eXPszPAF4PrNt" +
  "PGN4QSxAPX12DjvmT6CckFu5+dt72Th6eMzx6Up2pFlGPvV2KRJT/91HZwCUIBVFrItCMvxZpu3M" +
  "VrJomqeg5lQqX2ygovhOqnbroa6RXtK466VjtRrqXcigm79oUo7w9gEvFhWF";

function base64ToUint8Array(b64: string): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; ++i) out[i] = binary.charCodeAt(i);
  return out;
}

// Builds a minimal version-1 MapBuffer with a single (label -> blob) entry.
function buildMapBuffer(
  label: bigint,
  blobWithCrc: Uint8Array,
  compression = "00br",
): ArrayBuffer {
  const numKeys = 1;
  const headerSize = 16;
  const indexSize = numKeys * 16;
  const dataStart = headerSize + indexSize;
  const buffer = new ArrayBuffer(dataStart + blobWithCrc.length);
  const bytes = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  // magic
  for (let i = 0; i < 7; ++i) bytes[i] = "mapbufr".charCodeAt(i);
  bytes[7] = 1; // version
  for (let i = 0; i < 4; ++i) bytes[8 + i] = compression.charCodeAt(i);
  dv.setUint32(12, numKeys, /*littleEndian=*/ true);
  // index entry: label, absolute offset
  dv.setBigUint64(16, label, true);
  dv.setBigUint64(24, BigInt(dataStart), true);
  bytes.set(blobWithCrc, dataStart);
  return buffer;
}

describe("parseMapBuffer", () => {
  const label = 720575940566892690n;
  const blob = base64ToUint8Array(REAL_BLOB_BASE64);

  it("parses the header, index, and strips the v1 crc32c", () => {
    const parsed = parseMapBuffer(buildMapBuffer(label, blob));
    expect(parsed.version).toBe(1);
    expect(parsed.compression).toBe("00br");
    expect(parsed.entries).toHaveLength(1);
    expect(parsed.entries[0].label).toBe(label);
    // 273-byte blob minus the 4-byte crc32c.
    expect(parsed.entries[0].data.length).toBe(blob.length - 4);
  });

  it("decodes the brotli blob to a valid precomputed skeleton", () => {
    const parsed = parseMapBuffer(buildMapBuffer(label, blob));
    const decoded = decodeBrotli(parsed.entries[0].data);
    expect(decoded.length).toBe(616);
    const dv = new DataView(decoded.buffer, decoded.byteOffset, decoded.length);
    const numVertices = dv.getUint32(0, true);
    const numEdges = dv.getUint32(4, true);
    expect(numVertices).toBe(22);
    expect(numEdges).toBe(21);
    // 8-byte header + 12 B/vertex positions + 8 B/vertex (radius+csa) + 8 B/edge.
    expect(8 + numVertices * 20 + numEdges * 8).toBe(616);
  });

  it("rejects a buffer with a bad magic", () => {
    const bad = new ArrayBuffer(16);
    expect(() => parseMapBuffer(bad)).toThrow(/magic/);
  });

  it("handles an empty (zero-key) MapBuffer", () => {
    const buffer = new ArrayBuffer(16);
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < 7; ++i) bytes[i] = "mapbufr".charCodeAt(i);
    bytes[7] = 1;
    for (let i = 0; i < 4; ++i) bytes[8 + i] = "00br".charCodeAt(i);
    // numKeys stays 0.
    const parsed = parseMapBuffer(buffer);
    expect(parsed.entries).toHaveLength(0);
  });
});
