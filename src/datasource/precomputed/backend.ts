/**
 * @license
 * Copyright 2016 Google Inc.
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

import type {
  AnnotationGeometryChunk,
  AnnotationMetadataChunk,
  AnnotationSubsetGeometryChunk,
} from "#src/annotation/backend.js";
import {
  AnnotationGeometryData,
  AnnotationSource,
  AnnotationGeometryChunkSourceBackend,
} from "#src/annotation/backend.js";
import type { Annotation } from "#src/annotation/index.js";
import {
  AnnotationPropertySerializer,
  AnnotationType,
  annotationTypeHandlers,
  annotationTypes,
} from "#src/annotation/index.js";
import { WithParameters } from "#src/chunk_manager/backend.js";
import {
  AnnotationSourceParameters,
  AnnotationSpatialIndexSourceParameters,
  MeshSourceParameters,
  MultiscaleMeshSourceParameters,
  PrecomputedSpatialSkeletonSourceParameters,
  SkeletonSourceParameters,
  VolumeChunkEncoding,
  VolumeChunkSourceParameters,
} from "#src/datasource/precomputed/base.js";
import { parseMapBuffer } from "#src/datasource/precomputed/mapbuffer.js";
import type {
  ShardedKvStore,
  ShardInfo,
} from "#src/datasource/precomputed/sharded.js";
import { getShardedKvStoreIfApplicable } from "#src/datasource/precomputed/sharded.js";
import { WithSharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import type { KvStoreWithPath, ReadResponse } from "#src/kvstore/index.js";
import { readKvStore } from "#src/kvstore/index.js";
import type {
  FragmentChunk,
  ManifestChunk,
  MultiscaleFragmentChunk,
  MultiscaleManifestChunk,
} from "#src/mesh/backend.js";
import {
  assignMeshFragmentData,
  assignMultiscaleMeshFragmentData,
  computeOctreeChildOffsets,
  decodeJsonManifestChunk,
  decodeTriangleVertexPositionsAndIndices,
  decodeVertexPositionsAndIndices,
  generateHigherOctreeLevel,
  MeshSource,
  MultiscaleMeshSource,
} from "#src/mesh/backend.js";
import { decodeDracoPartitioned } from "#src/mesh/draco/index.js";
import type {
  SkeletonChunk,
  SpatiallyIndexedSkeletonChunk,
} from "#src/skeleton/backend.js";
import {
  SkeletonSource,
  SpatiallyIndexedSkeletonSourceBackend,
} from "#src/skeleton/backend.js";
import type { VertexAttributeInfo } from "#src/skeleton/base.js";
import { decodeSkeletonChunk } from "#src/skeleton/decode_precomputed_skeleton.js";
import { decodeCompressedSegmentationChunk } from "#src/sliceview/backend_chunk_decoders/compressed_segmentation.js";
import { decodeCompressoChunk } from "#src/sliceview/backend_chunk_decoders/compresso.js";
import type { ChunkDecoder } from "#src/sliceview/backend_chunk_decoders/index.js";
import { decodeJpegChunk } from "#src/sliceview/backend_chunk_decoders/jpeg.js";
import { decodeJxlChunk } from "#src/sliceview/backend_chunk_decoders/jxl.js";
import { decodePngChunk } from "#src/sliceview/backend_chunk_decoders/png.js";
import { decodeRawChunk } from "#src/sliceview/backend_chunk_decoders/raw.js";
import type { VolumeChunk } from "#src/sliceview/volume/backend.js";
import { VolumeChunkSource } from "#src/sliceview/volume/backend.js";
import { decodeBrotli } from "#src/util/brotli.js";
import { DATA_TYPE_BYTES } from "#src/util/data_type.js";
import {
  convertEndian16,
  convertEndian32,
  Endianness,
} from "#src/util/endian.js";
import { vec3 } from "#src/util/geom.js";
import { maybeDecompressGzip } from "#src/util/gzip.js";
import {
  encodeZIndexCompressed,
  encodeZIndexCompressed3d,
  zorder3LessThan,
} from "#src/util/zorder.js";
import { registerSharedObject } from "#src/worker_rpc.js";

// Set to true to validate the multiscale index.
const DEBUG_MULTISCALE_INDEX = false;

function getOrNotFoundError<T>(v: T | undefined) {
  if (v === undefined) throw new Error("not found");
  return v;
}

const chunkDecoders = new Map<VolumeChunkEncoding, ChunkDecoder>();
chunkDecoders.set(VolumeChunkEncoding.RAW, decodeRawChunk);
chunkDecoders.set(VolumeChunkEncoding.JPEG, decodeJpegChunk);
chunkDecoders.set(
  VolumeChunkEncoding.COMPRESSED_SEGMENTATION,
  decodeCompressedSegmentationChunk,
);
chunkDecoders.set(VolumeChunkEncoding.COMPRESSO, decodeCompressoChunk);
chunkDecoders.set(VolumeChunkEncoding.PNG, decodePngChunk);
chunkDecoders.set(VolumeChunkEncoding.JXL, decodeJxlChunk);

@registerSharedObject()
export class PrecomputedVolumeChunkSource extends WithParameters(
  WithSharedKvStoreContextCounterpart(VolumeChunkSource),
  VolumeChunkSourceParameters,
) {
  chunkDecoder = chunkDecoders.get(this.parameters.encoding)!;
  kvStore = this.sharedKvStoreContext.kvStoreContext.getKvStore(
    this.parameters.url,
  );
  shardedKvStore = getShardedKvStoreIfApplicable(
    this,
    this.kvStore,
    this.parameters.sharding,
  );

  gridShape = (() => {
    const gridShape = new Uint32Array(3);
    const { upperVoxelBound, chunkDataSize } = this.spec;
    for (let i = 0; i < 3; ++i) {
      gridShape[i] = Math.ceil(upperVoxelBound[i] / chunkDataSize[i]);
    }
    return gridShape;
  })();

  async download(chunk: VolumeChunk, signal: AbortSignal): Promise<void> {
    const { shardedKvStore } = this;
    let readResponse: ReadResponse | undefined;
    if (shardedKvStore === undefined) {
      const { kvStore } = this;
      let path: string;
      {
        // chunkPosition must not be captured, since it will be invalidated by the next call to
        // computeChunkBounds.
        const chunkPosition = this.computeChunkBounds(chunk);
        const chunkDataSize = chunk.chunkDataSize!;
        path =
          `${kvStore.path}${chunkPosition[0]}-${
            chunkPosition[0] + chunkDataSize[0]
          }_` +
          `${chunkPosition[1]}-${chunkPosition[1] + chunkDataSize[1]}_` +
          `${chunkPosition[2]}-${chunkPosition[2] + chunkDataSize[2]}`;
      }
      readResponse = await kvStore.store.read(path, { signal });
    } else {
      this.computeChunkBounds(chunk);
      const { gridShape } = this;
      const { chunkGridPosition } = chunk;
      const xBits = Math.ceil(Math.log2(gridShape[0]));
      const yBits = Math.ceil(Math.log2(gridShape[1]));
      const zBits = Math.ceil(Math.log2(gridShape[2]));
      const chunkIndex = encodeZIndexCompressed3d(
        xBits,
        yBits,
        zBits,
        chunkGridPosition[0],
        chunkGridPosition[1],
        chunkGridPosition[2],
      );
      readResponse = await shardedKvStore.read(chunkIndex, { signal });
    }
    if (readResponse !== undefined) {
      await this.chunkDecoder(
        chunk,
        signal,
        await readResponse.response.arrayBuffer(),
      );
    }
  }
}

export function decodeManifestChunk(chunk: ManifestChunk, response: any) {
  return decodeJsonManifestChunk(chunk, response, "fragments");
}

export function decodeFragmentChunk(
  chunk: FragmentChunk,
  response: ArrayBuffer,
) {
  const dv = new DataView(response);
  const numVertices = dv.getUint32(0, true);
  assignMeshFragmentData(
    chunk,
    decodeTriangleVertexPositionsAndIndices(
      response,
      Endianness.LITTLE,
      /*vertexByteOffset=*/ 4,
      numVertices,
    ),
  );
}

@registerSharedObject()
export class PrecomputedMeshSource extends WithParameters(
  WithSharedKvStoreContextCounterpart(MeshSource),
  MeshSourceParameters,
) {
  kvStore = this.sharedKvStoreContext.kvStoreContext.getKvStore(
    this.parameters.url,
  );
  async download(chunk: ManifestChunk, signal: AbortSignal) {
    const { parameters, kvStore } = this;
    const response = await readKvStore(
      kvStore.store,
      `${kvStore.path}${chunk.objectId}:${parameters.lod}`,
      { signal, throwIfMissing: true },
    );
    decodeManifestChunk(chunk, await response.response.json());
  }

  async downloadFragment(chunk: FragmentChunk, signal: AbortSignal) {
    const { kvStore } = this;
    const response = await readKvStore(
      kvStore.store,
      `${kvStore.path}${chunk.fragmentId}`,
      { signal, throwIfMissing: true },
    );
    decodeFragmentChunk(chunk, await response.response.arrayBuffer());
  }
}

interface PrecomputedMultiscaleManifestChunk extends MultiscaleManifestChunk {
  /**
   * Byte offsets into data file for each octree node.
   *
   * Stored as Float64Array to allow 53-bit integer values.
   */
  offsets: Float64Array;
  shardInfo?: ShardInfo;
}

function decodeMultiscaleManifestChunk(
  chunk: PrecomputedMultiscaleManifestChunk,
  response: ArrayBuffer,
) {
  if (response.byteLength < 28 || response.byteLength % 4 !== 0) {
    throw new Error(`Invalid index file size: ${response.byteLength}`);
  }
  const dv = new DataView(response);
  let offset = 0;
  const chunkShape = vec3.fromValues(
    dv.getFloat32(offset, /*littleEndian=*/ true),
    dv.getFloat32(offset + 4, /*littleEndian=*/ true),
    dv.getFloat32(offset + 8, /*littleEndian=*/ true),
  );
  offset += 12;
  const gridOrigin = vec3.fromValues(
    dv.getFloat32(offset, /*littleEndian=*/ true),
    dv.getFloat32(offset + 4, /*littleEndian=*/ true),
    dv.getFloat32(offset + 8, /*littleEndian=*/ true),
  );
  offset += 12;
  const numStoredLods = dv.getUint32(offset, /*littleEndian=*/ true);
  offset += 4;
  if (response.byteLength < offset + (4 + 4 + 4 * 3) * numStoredLods) {
    throw new Error(
      `Invalid index file size for ${numStoredLods} lods: ${response.byteLength}`,
    );
  }
  const storedLodScales = new Float32Array(response, offset, numStoredLods);
  offset += 4 * numStoredLods;
  convertEndian32(storedLodScales, Endianness.LITTLE);
  const vertexOffsets = new Float32Array(response, offset, numStoredLods * 3);
  convertEndian32(vertexOffsets, Endianness.LITTLE);
  offset += 12 * numStoredLods;
  const numFragmentsPerLod = new Uint32Array(response, offset, numStoredLods);
  offset += 4 * numStoredLods;
  convertEndian32(numFragmentsPerLod, Endianness.LITTLE);
  const totalFragments = numFragmentsPerLod.reduce((a, b) => a + b);
  if (response.byteLength !== offset + 16 * totalFragments) {
    throw new Error(
      `Invalid index file size for ${numStoredLods} lods and ` +
        `${totalFragments} total fragments: ${response.byteLength}`,
    );
  }
  const fragmentInfo = new Uint32Array(response, offset);
  convertEndian32(fragmentInfo, Endianness.LITTLE);
  const clipLowerBound = vec3.fromValues(
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  );
  const clipUpperBound = vec3.fromValues(
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  );
  let numLods = Math.max(1, storedLodScales.length);
  // Compute `clipLowerBound` and `clipUpperBound` and `numLods`.  Note that `numLods` is >=
  // `storedLodScales.length`; it may contain additional levels since at the highest level the
  // octree must be a single node.
  {
    let fragmentBase = 0;
    for (let lodIndex = 0; lodIndex < numStoredLods; ++lodIndex) {
      const numFragments = numFragmentsPerLod[lodIndex];
      if (DEBUG_MULTISCALE_INDEX) {
        for (let i = 1; i < numFragments; ++i) {
          const x0 = fragmentInfo[fragmentBase + numFragments * 0 + (i - 1)];
          const y0 = fragmentInfo[fragmentBase + numFragments * 1 + (i - 1)];
          const z0 = fragmentInfo[fragmentBase + numFragments * 2 + (i - 1)];
          const x1 = fragmentInfo[fragmentBase + numFragments * 0 + i];
          const y1 = fragmentInfo[fragmentBase + numFragments * 1 + i];
          const z1 = fragmentInfo[fragmentBase + numFragments * 2 + i];
          if (!zorder3LessThan(x0, y0, z0, x1, y1, z1)) {
            console.log(
              "Fragment index violates zorder constraint: " +
                `lod=${lodIndex}, ` +
                `chunk ${i - 1} = [${x0},${y0},${z0}], ` +
                `chunk ${i} = [${x1},${y1},${z1}]`,
            );
          }
        }
      }
      for (let i = 0; i < 3; ++i) {
        let upperBoundValue = Number.NEGATIVE_INFINITY;
        let lowerBoundValue = Number.POSITIVE_INFINITY;
        const base = fragmentBase + numFragments * i;
        for (let j = 0; j < numFragments; ++j) {
          const v = fragmentInfo[base + j];
          upperBoundValue = Math.max(upperBoundValue, v);
          lowerBoundValue = Math.min(lowerBoundValue, v);
        }
        if (numFragments !== 0) {
          while (
            upperBoundValue >>> (numLods - lodIndex - 1) !==
            lowerBoundValue >>> (numLods - lodIndex - 1)
          ) {
            ++numLods;
          }
          if (lodIndex === 0) {
            clipLowerBound[i] = Math.min(
              clipLowerBound[i],
              (1 << lodIndex) * lowerBoundValue,
            );
            clipUpperBound[i] = Math.max(
              clipUpperBound[i],
              (1 << lodIndex) * (upperBoundValue + 1),
            );
          }
        }
      }
      fragmentBase += numFragments * 4;
    }
  }

  // Compute upper bound on number of nodes that will be in the octree, so that we can allocate a
  // sufficiently large buffer without having to worry about resizing.
  let maxFragments = 0;
  {
    let prevNumFragments = 0;
    let prevLodIndex = 0;
    for (let lodIndex = 0; lodIndex < numStoredLods; ++lodIndex) {
      const numFragments = numFragmentsPerLod[lodIndex];
      maxFragments += prevNumFragments * (lodIndex - prevLodIndex);
      prevLodIndex = lodIndex;
      prevNumFragments = numFragments;
      maxFragments += numFragments;
    }
    maxFragments += (numLods - 1 - prevLodIndex) * prevNumFragments;
  }
  const octreeTemp = new Uint32Array(5 * maxFragments);
  const offsetsTemp = new Float64Array(maxFragments + 1);
  let octree: Uint32Array;
  {
    let priorStart = 0;
    let baseRow = 0;
    let dataOffset = 0;
    let fragmentBase = 0;
    for (let lodIndex = 0; lodIndex < numStoredLods; ++lodIndex) {
      const numFragments = numFragmentsPerLod[lodIndex];
      // Copy in indices
      for (let j = 0; j < numFragments; ++j) {
        for (let i = 0; i < 3; ++i) {
          octreeTemp[5 * (baseRow + j) + i] =
            fragmentInfo[fragmentBase + j + i * numFragments];
        }
        const dataSize = fragmentInfo[fragmentBase + j + 3 * numFragments];
        dataOffset += dataSize;
        offsetsTemp[baseRow + j + 1] = dataOffset;
        if (dataSize === 0) {
          // Mark node as empty.
          octreeTemp[5 * (baseRow + j) + 4] = 0x80000000;
        }
      }

      fragmentBase += 4 * numFragments;

      if (lodIndex !== 0) {
        // Connect with prior level
        computeOctreeChildOffsets(
          octreeTemp,
          priorStart,
          baseRow,
          baseRow + numFragments,
        );
      }

      priorStart = baseRow;
      baseRow += numFragments;
      while (
        lodIndex + 1 < numLods &&
        (lodIndex + 1 >= storedLodScales.length ||
          storedLodScales[lodIndex + 1] === 0)
      ) {
        const curEnd = generateHigherOctreeLevel(
          octreeTemp,
          priorStart,
          baseRow,
        );
        offsetsTemp.fill(dataOffset, baseRow + 1, curEnd + 1);
        priorStart = baseRow;
        baseRow = curEnd;
        ++lodIndex;
      }
    }
    octree = octreeTemp.slice(0, 5 * baseRow);
    chunk.offsets = offsetsTemp.slice(0, baseRow + 1);
  }
  const source = chunk.source! as PrecomputedMultiscaleMeshSource;
  const { lodScaleMultiplier } = source.parameters.metadata;
  const lodScales = new Float32Array(numLods);
  lodScales.set(storedLodScales, 0);
  for (let i = 0; i < storedLodScales.length; ++i) {
    lodScales[i] *= lodScaleMultiplier;
  }
  chunk.manifest = {
    chunkShape,
    chunkGridSpatialOrigin: gridOrigin,
    clipLowerBound: vec3.add(
      clipLowerBound,
      gridOrigin,
      vec3.multiply(clipLowerBound, clipLowerBound, chunkShape),
    ),
    clipUpperBound: vec3.add(
      clipUpperBound,
      gridOrigin,
      vec3.multiply(clipUpperBound, clipUpperBound, chunkShape),
    ),
    octree,
    lodScales,
    vertexOffsets,
  };
}

async function decodeMultiscaleFragmentChunk(
  chunk: MultiscaleFragmentChunk,
  response: ArrayBuffer,
) {
  const { lod } = chunk;
  const source = chunk.manifestChunk!
    .source! as PrecomputedMultiscaleMeshSource;
  const rawMesh = await decodeDracoPartitioned(
    new Uint8Array(response),
    source.parameters.metadata.vertexQuantizationBits,
    lod !== 0,
  );
  assignMultiscaleMeshFragmentData(
    chunk,
    rawMesh,
    source.format.vertexPositionFormat,
  );
}

@registerSharedObject() //
export class PrecomputedMultiscaleMeshSource extends WithParameters(
  WithSharedKvStoreContextCounterpart(MultiscaleMeshSource),
  MultiscaleMeshSourceParameters,
) {
  kvStore = this.sharedKvStoreContext.kvStoreContext.getKvStore(
    this.parameters.url,
  );
  shardedKvStore = getShardedKvStoreIfApplicable(
    this,
    this.kvStore,
    this.parameters.metadata.sharding,
  );

  async download(
    chunk: PrecomputedMultiscaleManifestChunk,
    signal: AbortSignal,
  ): Promise<void> {
    const { shardedKvStore } = this;
    let readResponse: ReadResponse | undefined;
    if (shardedKvStore === undefined) {
      const { kvStore } = this;
      readResponse = await kvStore.store.read(
        `${kvStore.path}${chunk.objectId}.index`,
        { signal },
      );
    } else {
      ({ response: readResponse, shardInfo: chunk.shardInfo } =
        getOrNotFoundError(
          await shardedKvStore.readWithShardInfo(chunk.objectId, {
            signal,
          }),
        ));
    }

    const data = await getOrNotFoundError(readResponse).response.arrayBuffer();

    decodeMultiscaleManifestChunk(chunk, data);
  }

  async downloadFragment(
    chunk: MultiscaleFragmentChunk,
    signal: AbortSignal,
  ): Promise<void> {
    const { kvStore } = this;
    const manifestChunk =
      chunk.manifestChunk! as PrecomputedMultiscaleManifestChunk;
    const chunkIndex = chunk.chunkIndex;
    const { shardInfo, offsets } = manifestChunk;
    const startOffset = offsets[chunkIndex];
    const endOffset = offsets[chunkIndex + 1];
    let requestPath: string;
    let adjustedStartOffset: number;
    let adjustedEndOffset: number;
    if (shardInfo !== undefined) {
      requestPath = shardInfo.shardPath;
      const fullDataSize = offsets[offsets.length - 1];
      const start = shardInfo.offset - fullDataSize + startOffset;
      const end = start + endOffset - startOffset;
      adjustedStartOffset = start;
      adjustedEndOffset = end;
    } else {
      requestPath = `${kvStore.path}${manifestChunk.objectId}`;
      adjustedStartOffset = startOffset;
      adjustedEndOffset = endOffset;
    }
    const readResponse = await readKvStore(kvStore.store, requestPath, {
      signal,
      byteRange: {
        offset: adjustedStartOffset,
        length: adjustedEndOffset - adjustedStartOffset,
      },
      throwIfMissing: true,
      strictByteRange: true,
    });
    await decodeMultiscaleFragmentChunk(
      chunk,
      await readResponse.response.arrayBuffer(),
    );
  }
}

async function fetchByUint64(
  chunkSource: {
    kvStore: KvStoreWithPath;
    shardedKvStore: ShardedKvStore | undefined;
  },
  id: bigint,
  signal: AbortSignal,
): Promise<ReadResponse | undefined> {
  const { shardedKvStore } = chunkSource;
  if (shardedKvStore === undefined) {
    const { kvStore } = chunkSource;
    return kvStore.store.read(`${kvStore.path}${id}`, {
      signal,
    });
  } else {
    return shardedKvStore.read(id, { signal });
  }
}

@registerSharedObject() //
export class PrecomputedSkeletonSource extends WithParameters(
  WithSharedKvStoreContextCounterpart(SkeletonSource),
  SkeletonSourceParameters,
) {
  kvStore = this.sharedKvStoreContext.kvStoreContext.getKvStore(
    this.parameters.url,
  );
  shardedKvStore = getShardedKvStoreIfApplicable(
    this,
    this.kvStore,
    this.parameters.metadata.sharding,
  );
  async download(chunk: SkeletonChunk, signal: AbortSignal) {
    const { parameters } = this;
    const response = getOrNotFoundError(
      await fetchByUint64(this, chunk.objectId, signal),
    );
    decodeSkeletonChunk(
      chunk,
      await response.response.arrayBuffer(),
      parameters.metadata.vertexAttributes,
    );
  }
}

// Decodes the vertex positions and edge indices of a single precomputed
// skeleton fragment (ignores vertex attributes such as radius for the spatial
// browse view).
// Returns an ArrayBuffer that starts exactly at the array's data (the fragment
// decoders index from offset 0).
function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer as ArrayBuffer;
  }
  return bytes.slice().buffer as ArrayBuffer;
}

function decodeSpatialSkeletonFragment(
  buffer: ArrayBuffer,
  vertexAttributes: Map<string, VertexAttributeInfo>,
) {
  const dv = new DataView(buffer);
  const numVertices = dv.getUint32(0, true);
  const numEdges = dv.getUint32(4, true);
  const { vertexPositions, indices } = decodeVertexPositionsAndIndices(
    /*verticesPerPrimitive=*/ 2,
    buffer,
    Endianness.LITTLE,
    /*vertexByteOffset=*/ 8,
    numVertices,
    /*indexByteOffset=*/ 8 + numVertices * 4 * 3,
    numEdges,
  );
  // Per-vertex attributes (e.g. radius, cross_sectional_area) follow the edges,
  // in the order declared by the info file. Returned as raw little-endian byte
  // blocks (one per attribute) to be concatenated and uploaded as-is.
  let offset = 8 + numVertices * 4 * 3 + numEdges * 4 * 2;
  const attributes: Uint8Array[] = [];
  for (const info of vertexAttributes.values()) {
    const bytesPerVertex = DATA_TYPE_BYTES[info.dataType] * info.numComponents;
    const totalBytes = bytesPerVertex * numVertices;
    const attribute = new Uint8Array(buffer, offset, totalBytes);
    switch (bytesPerVertex) {
      case 2:
        convertEndian16(attribute, Endianness.LITTLE);
        break;
      case 4:
      case 8:
        convertEndian32(attribute, Endianness.LITTLE);
        break;
    }
    attributes.push(attribute);
    offset += totalBytes;
  }
  return {
    numVertices,
    vertexPositions: vertexPositions as Float32Array,
    indices: indices as Uint32Array,
    attributes,
  };
}

// Runs `worker` over `items` with at most `limit` concurrent in flight.
async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  worker: (item: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  async function run() {
    while (true) {
      const index = next++;
      if (index >= items.length) return;
      results[index] = await worker(items[index]);
    }
  }
  const runners: Promise<void>[] = [];
  for (let i = 0; i < Math.min(limit, items.length); ++i) {
    runners.push(run());
  }
  await Promise.all(runners);
  return results;
}

type SpatialSkeletonFragment = ReturnType<
  typeof decodeSpatialSkeletonFragment
> & { id: bigint };

// Number of graph hops each side over which the per-vertex tangent and the
// rendered centerline positions are smoothed.
const TANGENT_SMOOTHING_HOPS = 3;

// Moving-average of vertex positions over the +/-`TANGENT_SMOOTHING_HOPS`-hop
// graph neighborhood, so the drawn polyline follows a smooth centerline (whose
// local direction matches the smoothed tangent) rather than the raw zig-zag.
function smoothFragmentPositions(
  numVertices: number,
  positions: Float32Array,
  indices: Uint32Array,
): Float32Array {
  const adjacency: number[][] = Array.from({ length: numVertices }, () => []);
  for (let e = 0; e < indices.length; e += 2) {
    const a = indices[e];
    const b = indices[e + 1];
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const out = new Float32Array(numVertices * 3);
  const visited = new Int32Array(numVertices).fill(-1);
  const queue = new Int32Array(numVertices);
  const depth = new Int32Array(numVertices);
  for (let v = 0; v < numVertices; ++v) {
    let sx = 0;
    let sy = 0;
    let sz = 0;
    let count = 0;
    let head = 0;
    let tail = 0;
    visited[v] = v;
    depth[v] = 0;
    queue[tail++] = v;
    while (head < tail) {
      const u = queue[head++];
      sx += positions[u * 3];
      sy += positions[u * 3 + 1];
      sz += positions[u * 3 + 2];
      ++count;
      if (depth[u] < TANGENT_SMOOTHING_HOPS) {
        for (const n of adjacency[u]) {
          if (visited[n] !== v) {
            visited[n] = v;
            depth[n] = depth[u] + 1;
            queue[tail++] = n;
          }
        }
      }
    }
    out[v * 3] = sx / count;
    out[v * 3 + 1] = sy / count;
    out[v * 3 + 2] = sz / count;
  }
  return out;
}

// Per-vertex tangent for a fragment, used for directional coloring of edges and
// nodes. Computed as the unit average of incident edge directions, then
// smoothed over a +/-`TANGENT_SMOOTHING_HOPS`-vertex graph neighborhood. Sign
// is arbitrary (consumers use abs()) — neighbor tangents are sign-aligned to the
// center before averaging so they don't cancel. Direction-less vertices get
// (0,0,0).
function computeFragmentTangents(
  numVertices: number,
  positions: Float32Array,
  indices: Uint32Array,
): Float32Array {
  // 1. Base per-vertex tangent = unit average of incident edge directions.
  const base = new Float32Array(numVertices * 3);
  for (let e = 0; e < indices.length; e += 2) {
    const a = indices[e];
    const b = indices[e + 1];
    let dx = positions[b * 3] - positions[a * 3];
    let dy = positions[b * 3 + 1] - positions[a * 3 + 1];
    let dz = positions[b * 3 + 2] - positions[a * 3 + 2];
    const len = Math.hypot(dx, dy, dz);
    if (len > 0) {
      dx /= len;
      dy /= len;
      dz /= len;
    }
    base[a * 3] += dx;
    base[a * 3 + 1] += dy;
    base[a * 3 + 2] += dz;
    base[b * 3] += dx;
    base[b * 3 + 1] += dy;
    base[b * 3 + 2] += dz;
  }
  for (let v = 0; v < numVertices; ++v) {
    const o = v * 3;
    const len = Math.hypot(base[o], base[o + 1], base[o + 2]);
    if (len > 0) {
      base[o] /= len;
      base[o + 1] /= len;
      base[o + 2] /= len;
    }
  }

  // 2. Adjacency.
  const adjacency: number[][] = Array.from({ length: numVertices }, () => []);
  for (let e = 0; e < indices.length; e += 2) {
    const a = indices[e];
    const b = indices[e + 1];
    adjacency[a].push(b);
    adjacency[b].push(a);
  }

  // 3. Smooth: average base tangents over the graph neighborhood within
  // `TANGENT_SMOOTHING_HOPS` hops (BFS), sign-aligned to the center vertex.
  const out = new Float32Array(numVertices * 3);
  // `visited` uses the center vertex index as a per-BFS generation marker to
  // avoid reallocating; `queue`/`depth` are scratch reused across vertices.
  const visited = new Int32Array(numVertices).fill(-1);
  const queue = new Int32Array(numVertices);
  const depth = new Int32Array(numVertices);
  for (let v = 0; v < numVertices; ++v) {
    const cx = base[v * 3];
    const cy = base[v * 3 + 1];
    const cz = base[v * 3 + 2];
    let sx = 0;
    let sy = 0;
    let sz = 0;
    let head = 0;
    let tail = 0;
    visited[v] = v;
    depth[v] = 0;
    queue[tail++] = v;
    while (head < tail) {
      const u = queue[head++];
      const ux = base[u * 3];
      const uy = base[u * 3 + 1];
      const uz = base[u * 3 + 2];
      const sign = ux * cx + uy * cy + uz * cz >= 0 ? 1 : -1;
      sx += sign * ux;
      sy += sign * uy;
      sz += sign * uz;
      if (depth[u] < TANGENT_SMOOTHING_HOPS) {
        for (const n of adjacency[u]) {
          if (visited[n] !== v) {
            visited[n] = v;
            depth[n] = depth[u] + 1;
            queue[tail++] = n;
          }
        }
      }
    }
    const len = Math.hypot(sx, sy, sz);
    if (len > 0) {
      out[v * 3] = sx / len;
      out[v * 3 + 1] = sy / len;
      out[v * 3 + 2] = sz / len;
    }
  }
  return out;
}

// Concatenates decoded fragments into a single chunk geometry: positions are
// shifted into the gridOrigin-relative nm frame the chunk grid uses, edges are
// re-based by the running vertex count, the per-vertex segment-id (uint64)
// attribute is filled with each fragment's label, a synthesized per-vertex
// `tangent` (vec3) is computed for directional coloring, and the per-vertex
// info attributes (e.g. radius, cross_sectional_area) are concatenated as raw
// bytes. The resulting `vertexAttributes` order
// (`[segment, tangent, ...infoAttributes]`) matches the frontend source's
// declared `vertexAttributes` (minus the implicit position at slot 0).
function packSpatialSkeletonFragments(
  chunk: SpatiallyIndexedSkeletonChunk,
  fragments: ReadonlyArray<SpatialSkeletonFragment | undefined>,
  vertexAttributes: Map<string, VertexAttributeInfo>,
  gridOrigin: Float32Array,
) {
  const attributeBytesPerVertex = Array.from(
    vertexAttributes.values(),
    (info) => DATA_TYPE_BYTES[info.dataType] * info.numComponents,
  );

  let totalVertices = 0;
  let totalIndices = 0;
  for (const fragment of fragments) {
    if (fragment === undefined) continue;
    totalVertices += fragment.numVertices;
    totalIndices += fragment.indices.length;
  }

  const vertexPositions = new Float32Array(totalVertices * 3);
  const segmentIdAttribute = new BigUint64Array(totalVertices);
  const tangentAttribute = new Float32Array(totalVertices * 3);
  const indices = new Uint32Array(totalIndices);
  const infoAttributes = attributeBytesPerVertex.map(
    (bytesPerVertex) => new Uint8Array(bytesPerVertex * totalVertices),
  );
  let vertexOffset = 0;
  let indexOffset = 0;
  for (const fragment of fragments) {
    if (fragment === undefined) continue;
    const { numVertices, vertexPositions: rawPositions } = fragment;
    // Smooth the centerline so the drawn lines follow the smoothed direction,
    // and derive the tangent from the same smoothed path so color and geometry
    // are consistent.
    const fragPositions = smoothFragmentPositions(
      numVertices,
      rawPositions,
      fragment.indices,
    );
    const fragTangents = computeFragmentTangents(
      numVertices,
      fragPositions,
      fragment.indices,
    );
    tangentAttribute.set(fragTangents, vertexOffset * 3);
    for (let v = 0; v < numVertices; ++v) {
      const src = v * 3;
      const dst = (vertexOffset + v) * 3;
      vertexPositions[dst] = fragPositions[src] - gridOrigin[0];
      vertexPositions[dst + 1] = fragPositions[src + 1] - gridOrigin[1];
      vertexPositions[dst + 2] = fragPositions[src + 2] - gridOrigin[2];
      segmentIdAttribute[vertexOffset + v] = fragment.id;
    }
    for (let a = 0; a < infoAttributes.length; ++a) {
      const bytesPerVertex = attributeBytesPerVertex[a];
      infoAttributes[a].set(
        fragment.attributes[a],
        vertexOffset * bytesPerVertex,
      );
    }
    const { indices: fragIndices } = fragment;
    for (let e = 0; e < fragIndices.length; ++e) {
      indices[indexOffset + e] = fragIndices[e] + vertexOffset;
    }
    vertexOffset += numVertices;
    indexOffset += fragIndices.length;
  }

  chunk.vertexPositions = vertexPositions;
  chunk.indices = indices;
  chunk.vertexAttributes = [
    segmentIdAttribute,
    tangentAttribute,
    ...infoAttributes,
  ];
}

// Decompresses one MapBuffer blob according to the file's declared compression.
async function decompressMapBufferBlob(
  compression: string,
  data: Uint8Array<ArrayBuffer>,
): Promise<Uint8Array> {
  switch (compression) {
    case "none":
      return data;
    case "00br": // brotli
      return decodeBrotli(data);
    case "gzip":
      return maybeDecompressGzip(data);
    default:
      // zstd/lzma are valid MapBuffer codecs but not yet wired here.
      throw new Error(`Unsupported .frags compression: ${compression}`);
  }
}

@registerSharedObject()
export class PrecomputedSpatialSkeletonSourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(SpatiallyIndexedSkeletonSourceBackend),
  PrecomputedSpatialSkeletonSourceParameters,
) {
  kvStore = this.sharedKvStoreContext.kvStoreContext.getKvStore(
    this.parameters.url,
  );
  shardedKvStore = getShardedKvStoreIfApplicable(
    this,
    this.kvStore,
    this.parameters.metadata.sharding,
  );
  // Set once we determine this source has no `.frags` files (a `.frags` miss
  // accompanied by a `.spatial` hit), so subsequent chunks skip the `.frags`
  // probe and go straight to the sharded fallback.
  private fragsUnavailable = false;

  async download(chunk: SpatiallyIndexedSkeletonChunk, signal: AbortSignal) {
    const { metadata, gridOrigin } = this.parameters;
    const { chunkGridPosition } = chunk;

    // Primary path: the per-chunk `.frags` MapBuffer (one request for every
    // skeleton in the chunk).
    if (!this.fragsUnavailable) {
      const fragments = await this.downloadFrags(chunkGridPosition, signal);
      if (fragments !== undefined) {
        packSpatialSkeletonFragments(
          chunk,
          fragments,
          metadata.vertexAttributes,
          gridOrigin,
        );
        return;
      }
    }

    // Fallback: `.spatial` index + per-segment fragments from the (sharded)
    // store.
    await this.downloadFromSpatial(chunk, chunkGridPosition, signal);
  }

  // Reads and decodes the `.frags` MapBuffer for a chunk. Returns the decoded
  // fragments, or `undefined` if the `.frags` file is absent (caller falls
  // back). An empty-but-present `.frags` returns `[]`.
  private async downloadFrags(
    chunkGridPosition: Float32Array,
    signal: AbortSignal,
  ): Promise<SpatialSkeletonFragment[] | undefined> {
    const { metadata, gridOrigin } = this.parameters;
    const { chunkSize, resolution } = metadata.spatialIndex!;
    // `.frags` filenames are in voxels at the skeleton resolution.
    const lower = new Array<number>(3);
    const upper = new Array<number>(3);
    for (let i = 0; i < 3; ++i) {
      const chunkVox = Math.round(chunkSize[i] / resolution[i]);
      const originVox = Math.round(gridOrigin[i] / resolution[i]);
      lower[i] = originVox + chunkGridPosition[i] * chunkVox;
      upper[i] = lower[i] + chunkVox;
    }
    const name =
      `${lower[0]}-${upper[0]}_` +
      `${lower[1]}-${upper[1]}_` +
      `${lower[2]}-${upper[2]}.frags`;
    const response = await this.kvStore.store.read(
      `${this.kvStore.path}${name}`,
      { signal },
    );
    if (response === undefined) return undefined;
    const { compression, entries } = parseMapBuffer(
      await response.response.arrayBuffer(),
    );
    const fragments: SpatialSkeletonFragment[] = [];
    for (const entry of entries) {
      const skeletonBytes = await decompressMapBufferBlob(
        compression,
        entry.data,
      );
      fragments.push({
        id: entry.label,
        ...decodeSpatialSkeletonFragment(
          toArrayBuffer(skeletonBytes),
          metadata.vertexAttributes,
        ),
      });
    }
    return fragments;
  }

  // Reads the `.spatial` JSON index (nm-named) and fetches each listed skeleton
  // fragment. For sharded stores, `readBatch` coalesces what would be one HTTP
  // range request per segment into a few per-minishard reads.
  private async downloadFromSpatial(
    chunk: SpatiallyIndexedSkeletonChunk,
    chunkGridPosition: Float32Array,
    signal: AbortSignal,
  ) {
    const { metadata, gridOrigin } = this.parameters;
    const chunkSize = metadata.spatialIndex!.chunkSize;

    // Absolute (nanometer) bounding box of this chunk; the `.spatial` files are
    // named by their nm bbox on a grid offset by `gridOrigin`.
    const lower = new Float64Array(3);
    for (let i = 0; i < 3; ++i) {
      lower[i] = gridOrigin[i] + chunkGridPosition[i] * chunkSize[i];
    }
    const name =
      `${lower[0]}-${lower[0] + chunkSize[0]}_` +
      `${lower[1]}-${lower[1] + chunkSize[1]}_` +
      `${lower[2]}-${lower[2] + chunkSize[2]}.spatial`;

    const indexResponse = await this.kvStore.store.read(
      `${this.kvStore.path}${name}`,
      { signal },
    );
    if (indexResponse === undefined) {
      // Neither `.frags` nor `.spatial` for this chunk: empty.
      packSpatialSkeletonFragments(
        chunk,
        [],
        metadata.vertexAttributes,
        gridOrigin,
      );
      return;
    }
    // `.spatial` exists but `.frags` did not: this source uses the sharded
    // layout, so skip the `.frags` probe for future chunks.
    this.fragsUnavailable = true;

    const indexBytes = await maybeDecompressGzip(
      await indexResponse.response.arrayBuffer(),
    );
    const indexJson = JSON.parse(new TextDecoder().decode(indexBytes));
    const segmentIds = Object.keys(indexJson).map((k) => BigInt(k));
    if (segmentIds.length === 0) {
      packSpatialSkeletonFragments(
        chunk,
        [],
        metadata.vertexAttributes,
        gridOrigin,
      );
      return;
    }

    const { shardedKvStore } = this;
    let fragments: Array<SpatialSkeletonFragment | undefined>;
    if (shardedKvStore !== undefined) {
      const dataMap = await shardedKvStore.readBatch(segmentIds, { signal });
      fragments = segmentIds.map((id) => {
        const bytes = dataMap.get(id);
        if (bytes === undefined) return undefined;
        return {
          id,
          ...decodeSpatialSkeletonFragment(
            toArrayBuffer(bytes),
            metadata.vertexAttributes,
          ),
        };
      });
    } else {
      fragments = await mapWithConcurrency(segmentIds, 16, async (id) => {
        const response = await fetchByUint64(this, id, signal);
        if (response === undefined) return undefined;
        return {
          id,
          ...decodeSpatialSkeletonFragment(
            await response.response.arrayBuffer(),
            metadata.vertexAttributes,
          ),
        };
      });
    }

    packSpatialSkeletonFragments(
      chunk,
      fragments,
      metadata.vertexAttributes,
      gridOrigin,
    );
  }
}

function parseAnnotations(
  buffer: ArrayBuffer,
  parameters: AnnotationSourceParameters,
  propertySerializer: AnnotationPropertySerializer,
): AnnotationGeometryData {
  // FIXME: convert endian in order to support big endian platforms
  const isLittleEndian = true;
  // First, compute simple sanity checks for sizes etc. to verify that the buffer is well-formed.
  if (buffer.byteLength < 8) throw new Error("Expected at least 8 bytes");
  const dv = new DataView(buffer);
  const countHigh = dv.getUint32(4, isLittleEndian);
  if (countHigh !== 0) throw new Error("Annotation count too high");
  const numAnnotations = dv.getUint32(0, isLittleEndian);

  // Compute the size of the input buffer
  const numBytesPerInstance = propertySerializer.serializedBytes;
  let expectedNonIndexInputBytes = 8 + numBytesPerInstance * numAnnotations;
  let numInstances = numAnnotations;

  // If the annotation type is a polyline, we need to compute the number of instances
  const annotationType = parameters.type;
  if (annotationType === AnnotationType.POLYLINE) {
    const result = calculatePolylineMemoryUsage(
      dv,
      parameters.rank,
      numAnnotations,
      numBytesPerInstance,
      isLittleEndian,
    );
    numInstances = result.numInstances;
    expectedNonIndexInputBytes = result.totalBytes;
  }

  // Get the unique uint64 ids as string that sit at the end of the buffer
  const ids = extractAnnotationIdsFromBuffer(
    buffer,
    dv,
    numAnnotations,
    expectedNonIndexInputBytes,
  );

  // Now, create the geometry and properties data object
  // Polylines and data with lots of properties need to be reformatted
  const inputData = new Uint8Array(buffer, 8, expectedNonIndexInputBytes - 8);
  const geometryData = new AnnotationGeometryData();
  const typeToInstanceCounts = (geometryData.typeToInstanceCounts = new Array<
    number[]
  >(annotationTypes.length));
  typeToInstanceCounts.fill([]);
  const { propertyGroupBytes } = propertySerializer;
  if (
    propertyGroupBytes.length > 1 ||
    annotationType === AnnotationType.POLYLINE
  ) {
    const result = restructureAnnotationData(
      dv,
      inputData,
      propertySerializer,
      numInstances,
      numAnnotations,
      annotationType,
      parameters.rank,
      isLittleEndian,
    );
    geometryData.data = result.outputData;
    typeToInstanceCounts[AnnotationType.POLYLINE] =
      result.polylineInstanceCounts;
  } else {
    geometryData.data = inputData;
    typeToInstanceCounts[parameters.type] = Array.from(
      { length: ids.length },
      (_, i) => i,
    );
  }

  // Fill in the rest of the required geometry data
  const typeToOffset = (geometryData.typeToOffset = new Array<number>(
    annotationTypes.length,
  ));
  typeToOffset.fill(0);
  const typeToIds = (geometryData.typeToIds = new Array<string[]>(
    annotationTypes.length,
  ));
  const typeToIdMaps = (geometryData.typeToIdMaps = new Array<
    Map<string, number>
  >(annotationTypes.length));
  const typeToSize = (geometryData.typeToSize = new Array<number>(
    annotationTypes.length,
  ));
  typeToSize.fill(0);
  typeToSize[parameters.type] = numInstances;
  typeToIds.fill([]);
  typeToIds[parameters.type] = ids;
  typeToIdMaps.fill(new Map());
  typeToIdMaps[parameters.type] = new Map(ids.map((id, i) => [id, i]));
  return geometryData;
}

/**
 * Rearranges the annotation data into a format that is more suitable for WebGL.
 *
 * This is needed for annotations with multiple properties, and for polylines.
 */
function restructureAnnotationData(
  inputDataView: DataView<ArrayBuffer>,
  inputData: Uint8Array<ArrayBuffer>,
  propertySerializer: AnnotationPropertySerializer,
  numInstances: number,
  numAnnotations: number,
  annotationType: AnnotationType,
  rank: number,
  isLittleEndian: boolean,
) {
  const { propertyGroupBytes, serializedBytes: numBytesPerInstance } =
    propertySerializer;
  const glBufferSize = numBytesPerInstance * numInstances;
  const outputData = new Uint8Array(glBufferSize);
  let polylineInstanceCounts: number[] = [];

  if (annotationType === AnnotationType.POLYLINE) {
    polylineInstanceCounts = reformatPolylineBuffer(
      inputDataView,
      inputData,
      outputData,
      propertySerializer,
      numAnnotations,
      rank,
      isLittleEndian,
    );
  }

  // Places all the properties that can't be put in the first group into their own
  // group at the end of the buffer
  let dataToTransform = inputData;
  if (annotationType === AnnotationType.POLYLINE) {
    dataToTransform = new Uint8Array(outputData);
  }
  if (propertyGroupBytes.length > 1) {
    let origOffset = 0;
    let groupOffset = 0;
    for (
      let groupIndex = 0;
      groupIndex < propertyGroupBytes.length;
      ++groupIndex
    ) {
      let runningTotalInstances = 0;
      const groupBytesPerAnnotation = propertyGroupBytes[groupIndex];
      for (
        let annotationIndex = 0;
        annotationIndex < numAnnotations;
        ++annotationIndex
      ) {
        let numGlInstances = 1;
        if (annotationType === AnnotationType.POLYLINE) {
          // Use the polyline instance count to get the number of instances
          if (annotationIndex === numAnnotations - 1) {
            numGlInstances =
              numInstances - polylineInstanceCounts[annotationIndex];
          } else {
            numGlInstances =
              polylineInstanceCounts[annotationIndex + 1] -
              polylineInstanceCounts[annotationIndex];
          }
        }
        for (
          let instanceIndex = 0;
          instanceIndex < numGlInstances;
          ++instanceIndex
        ) {
          const origBase =
            origOffset + runningTotalInstances * numBytesPerInstance;
          const newBase =
            groupOffset + runningTotalInstances * groupBytesPerAnnotation;
          outputData.set(
            dataToTransform.subarray(
              origBase,
              origBase + groupBytesPerAnnotation,
            ),
            newBase,
          );
          ++runningTotalInstances;
        }
      }
      origOffset += groupBytesPerAnnotation;
      groupOffset += groupBytesPerAnnotation * numInstances;
    }
  }
  return { outputData, polylineInstanceCounts };
}

/**
 * Reformats the polyline buffer into a format that is more suitable for WebGL.
 *
 * The input data layout is:
 *
 * NumPointsInPoly1, Point1_1, Point1_2, ..., Point1_N1, Properties1,
 * NumPointsInPoly2, Point2_1, Point2_2, ..., Point2_N2, Properties2,
 * ...,
 * NumPointsInPolyK, PointK_1, PointK_2, ..., PointK_Nk, PropertiesK
 *
 * While the GL buffer data layout is:
 * PolyLineIndex1_1, Point1_1, Point1_2, Properties1,
 * PolyLineIndex1_2, Point1_2, Point1_3, Properties1,
 * ...,
 * PolyLineIndex1_N-1, Point1_N1-1, Point1_N1, Properties1,
 * ...,
 * PolyLineIndexK_1, PointK_1, PointK_2, PropertiesK
 */
function reformatPolylineBuffer(
  inputDataView: DataView<ArrayBuffer>,
  inputData: Uint8Array<ArrayBuffer>,
  outputData: Uint8Array<ArrayBuffer>,
  propertySerializer: AnnotationPropertySerializer,
  numAnnotations: number,
  rank: number,
  isLittleEndian: boolean,
): number[] {
  const outputDataView = new DataView(outputData.buffer);

  let inputDataOffset = 0;
  let outputDataOffset = 0;
  const pointCountBytes = 4;
  const pointBytes = rank * 4;
  const numBytesPerInstance = propertySerializer.serializedBytes;
  const numPropertyBytes =
    numBytesPerInstance - 2 * pointBytes - pointCountBytes;
  const numAnnotationsOffset = 8;
  let cumulativeInstances = 0;
  const instanceCounts = new Array<number>(numAnnotations);

  for (let i = 0; i < numAnnotations; ++i) {
    const numPoints = inputDataView.getUint32(
      inputDataOffset + numAnnotationsOffset,
      isLittleEndian,
    );
    const numInstancesInAnnotation = numPoints - 1;
    inputDataOffset += pointCountBytes; // Move past the number of points
    const propertyDataStart = inputDataOffset + numPoints * pointBytes;
    instanceCounts[i] = cumulativeInstances;
    cumulativeInstances += numInstancesInAnnotation;

    for (let j = 0; j < numInstancesInAnnotation; ++j) {
      // First, we need to set the instance index for this annotation, where the
      // last bit is actually whether to draw the endpoint cap
      const bitCap = j === numInstancesInAnnotation - 1 ? 1 : 0;
      const instanceIndexWithBitCap = j | (bitCap << 31);
      outputDataView.setUint32(
        outputDataOffset,
        instanceIndexWithBitCap,
        isLittleEndian,
      );
      // Copy the geometry data for two points
      outputData.set(
        inputData.subarray(inputDataOffset, inputDataOffset + 2 * pointBytes),
        outputDataOffset + pointCountBytes,
      );
      // Copy the properties data
      outputData.set(
        inputData.subarray(
          propertyDataStart,
          propertyDataStart + numPropertyBytes,
        ),
        outputDataOffset + pointCountBytes + 2 * pointBytes,
      );

      inputDataOffset += pointBytes; // Move to the next point in the input data
      outputDataOffset += numBytesPerInstance; // Move to the next instance in the output data
    }
    inputDataOffset = propertyDataStart + numPropertyBytes;
  }
  return instanceCounts;
}

/**
 * Calculates the memory usage of a polyline annotation.
 *
 * Since the polyline annotation is stored in a different format, we need to
 * calculate the number of instances and the total memory usage.
 * Other annotation types are fixed size per annotation.
 */
function calculatePolylineMemoryUsage(
  dv: DataView<ArrayBuffer>,
  rank: number,
  numAnnotations: number,
  numBytesPerInstance: number,
  isLittleEndian: boolean,
) {
  let memoryOffset = 8; // Starting count of total number of annotations
  let numInstances = 0;
  for (let i = 0; i < numAnnotations; i++) {
    const numPoints = dv.getUint32(memoryOffset, isLittleEndian);
    const numGlInstances = numPoints - 1;
    const numGeometryBytes = numPoints * rank * 4;
    const numPropertyBytes = numBytesPerInstance - 2 * rank * 4;
    memoryOffset += numGeometryBytes + numPropertyBytes;
    numInstances += numGlInstances;
  }
  return { totalBytes: memoryOffset, numInstances };
}

function extractAnnotationIdsFromBuffer(
  buffer: ArrayBuffer,
  dv: DataView<ArrayBuffer>,
  numAnnotations: number,
  offset: number,
) {
  const expectedInputBytes = offset + 8 * numAnnotations;
  if (buffer.byteLength !== expectedInputBytes) {
    throw new Error(
      `Expected ${expectedInputBytes} bytes, but received: ${buffer.byteLength} bytes`,
    );
  }
  // Reading all of the ids at the end of the buffer
  const idOffset = offset;
  const ids = new Array<string>(numAnnotations);
  for (let i = 0; i < numAnnotations; ++i) {
    ids[i] = dv
      .getBigUint64(idOffset + i * 8, /*littleEndian=*/ true)
      .toString();
  }
  return ids;
}

function parseSingleAnnotation(
  buffer: ArrayBuffer,
  parameters: AnnotationSourceParameters,
  propertySerializer: AnnotationPropertySerializer,
  id: string,
): Annotation {
  const handler = annotationTypeHandlers[parameters.type];
  let baseNumBytes = propertySerializer.serializedBytes;
  const dv = new DataView(buffer);
  let offset = 0;
  if (parameters.type === AnnotationType.POLYLINE) {
    const numPolylinePoints =
      dv.getUint32(0, /*isLittleEndian=*/ true) & 0x7fffffff;
    const numPropertyBytes =
      propertySerializer.serializedBytes - (2 * 4 * parameters.rank + 4);
    baseNumBytes =
      4 + numPolylinePoints * 4 * parameters.rank + numPropertyBytes;
    offset = (numPolylinePoints - 2) * 4 * parameters.rank;
  }
  const numRelationships = parameters.relationships.length;
  const minNumBytes = baseNumBytes + 4 * numRelationships;
  if (buffer.byteLength < minNumBytes) {
    throw new Error(
      `Expected at least ${minNumBytes} bytes, but received: ${buffer.byteLength}`,
    );
  }
  const annotation = handler.deserialize(
    dv,
    0,
    /*isLittleEndian=*/ true,
    parameters.rank,
    id,
    0,
  );
  propertySerializer.deserialize(
    dv,
    offset,
    /*annotationIndex=*/ 0,
    /*annotationCount=*/ 1,
    /*isLittleEndian=*/ true,
    (annotation.properties = new Array(parameters.properties.length)),
  );
  offset = baseNumBytes;
  const relatedSegments: BigUint64Array[] = (annotation.relatedSegments = []);
  relatedSegments.length = numRelationships;
  for (let i = 0; i < numRelationships; ++i) {
    const count = dv.getUint32(offset, /*littleEndian=*/ true);
    if (buffer.byteLength < minNumBytes + count * 8) {
      throw new Error(
        `Expected at least ${minNumBytes} bytes, but received: ${buffer.byteLength}`,
      );
    }
    offset += 4;
    const segments = (relatedSegments[i] = new BigUint64Array(count));
    for (let j = 0; j < count; ++j) {
      segments[j] = dv.getBigUint64(offset, /*littleEndian=*/ true);
      offset += 8;
    }
  }
  if (offset !== buffer.byteLength) {
    throw new Error(
      `Expected ${offset} bytes, but received: ${buffer.byteLength}`,
    );
  }
  return annotation;
}

@registerSharedObject() //
export class PrecomputedAnnotationSpatialIndexSourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(AnnotationGeometryChunkSourceBackend),
  AnnotationSpatialIndexSourceParameters,
) {
  kvStore = this.sharedKvStoreContext.kvStoreContext.getKvStore(
    this.parameters.url,
  );
  shardedKvStore = getShardedKvStoreIfApplicable(
    this,
    this.kvStore,
    this.parameters.sharding,
  );
  declare parent: PrecomputedAnnotationSourceBackend;
  async download(chunk: AnnotationGeometryChunk, signal: AbortSignal) {
    const { shardedKvStore } = this;
    const { parent } = this;
    let response: ReadResponse | undefined;
    const { chunkGridPosition } = chunk;
    if (shardedKvStore === undefined) {
      const { kvStore } = this;
      const path = `${kvStore.path}${chunkGridPosition.join("_")}`;
      response = await kvStore.store.read(path, { signal });
    } else {
      const { upperChunkBound } = this.spec;
      const { chunkGridPosition } = chunk;
      const chunkIndex = encodeZIndexCompressed(
        chunkGridPosition,
        upperChunkBound,
      );
      response = await shardedKvStore.read(chunkIndex, { signal });
    }
    if (response !== undefined) {
      chunk.data = parseAnnotations(
        await response.response.arrayBuffer(),
        parent.parameters,
        parent.annotationPropertySerializer,
      );
    }
  }
}

@registerSharedObject() //
export class PrecomputedAnnotationSourceBackend extends WithParameters(
  WithSharedKvStoreContextCounterpart(AnnotationSource),
  AnnotationSourceParameters,
) {
  kvStore = this.sharedKvStoreContext.kvStoreContext.getKvStore(
    this.parameters.byId.url,
  );
  shardedKvStore = getShardedKvStoreIfApplicable(
    this,
    this.kvStore,
    this.parameters.byId.sharding,
  );
  private relationshipIndexSource = this.parameters.relationships.map((x) => {
    const kvStore = this.sharedKvStoreContext.kvStoreContext.getKvStore(x.url);
    const shardedKvStore = getShardedKvStoreIfApplicable(
      this,
      kvStore,
      x.sharding,
    );
    return { kvStore, shardedKvStore };
  });
  annotationPropertySerializer = new AnnotationPropertySerializer(
    this.parameters.rank,
    annotationTypeHandlers[this.parameters.type].serializedBytes(
      this.parameters.rank,
    ),
    this.parameters.properties,
  );

  async downloadSegmentFilteredGeometry(
    chunk: AnnotationSubsetGeometryChunk,
    relationshipIndex: number,
    signal: AbortSignal,
  ) {
    const response = await fetchByUint64(
      this.relationshipIndexSource[relationshipIndex],
      chunk.objectId,
      signal,
    );
    if (response !== undefined) {
      chunk.data = parseAnnotations(
        await response.response.arrayBuffer(),
        this.parameters,
        this.annotationPropertySerializer,
      );
    }
  }

  async downloadMetadata(chunk: AnnotationMetadataChunk, signal: AbortSignal) {
    const id = BigInt(chunk.key!);
    const response = await fetchByUint64(this, id, signal);
    if (response === undefined) {
      chunk.annotation = null;
    } else {
      chunk.annotation = parseSingleAnnotation(
        await response.response.arrayBuffer(),
        this.parameters,
        this.annotationPropertySerializer,
        chunk.key!,
      );
    }
  }
}
