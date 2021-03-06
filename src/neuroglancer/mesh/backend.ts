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

import debounce from 'lodash/debounce';
import {Chunk, ChunkSource} from 'neuroglancer/chunk_manager/backend';
import {ChunkPriorityTier, ChunkState} from 'neuroglancer/chunk_manager/base';
import {ChunkedGraphLayer} from 'neuroglancer/chunked_graph/backend';
import {FRAGMENT_SOURCE_RPC_ID, MESH_LAYER_RPC_ID} from 'neuroglancer/mesh/base';
import {SegmentationLayerSharedObjectCounterpart} from 'neuroglancer/segmentation_display_state/backend';
import {getObjectKey} from 'neuroglancer/segmentation_display_state/base';
import {forEachVisibleSegment3D, Bounds} from 'neuroglancer/segmentation_display_state/base';
import {CancellationToken} from 'neuroglancer/util/cancellation';
import {convertEndian32, Endianness} from 'neuroglancer/util/endian';
import {vec3} from 'neuroglancer/util/geom';
import {verifyObject, verifyObjectProperty, verifyStringArray} from 'neuroglancer/util/json';
import {Uint64} from 'neuroglancer/util/uint64';
import {getBasePriority, getPriorityTier} from 'neuroglancer/visibility_priority/backend';
import {registerSharedObject, RPC} from 'neuroglancer/worker_rpc';

const MESH_OBJECT_MANIFEST_CHUNK_PRIORITY = 100;
const MESH_OBJECT_FRAGMENT_CHUNK_PRIORITY = 50;

const DEBUG = false;

export type FragmentId = string;

// Chunk that contains the list of fragments that make up a single object.
export class ManifestChunk extends Chunk {
  objectId = new Uint64();
  fragmentIds: FragmentId[]|null;
  clipBounds?: Bounds;

  constructor() {
    super();
  }
  // We can't save a reference to objectId, because it may be a temporary
  // object.
  initializeManifestChunk(key: string, objectId: Uint64, clipBounds?: Bounds) {
    super.initialize(key);
    this.objectId.assign(objectId);
    if (clipBounds) {
      this.clipBounds = clipBounds;
    }
  }

  freeSystemMemory() {
    this.fragmentIds = null;
  }

  serialize(msg: any, transfers: any[]) {
    super.serialize(msg, transfers);
    msg.fragmentIds = this.fragmentIds;
  }

  downloadSucceeded() {
    // We can't easily determine the memory usage of the JSON manifest.  Just use 100 bytes as a
    // default value.
    this.systemMemoryBytes = 100;
    this.gpuMemoryBytes = 0;
    super.downloadSucceeded();
    if (this.priorityTier < ChunkPriorityTier.RECENT) {
      this.source!.chunkManager.scheduleUpdateChunkPriorities();
    }
  }

  downloadFailed(error: any) {
    // Missing manifest means remeshing is in progress. Initiate loading of child chunks.
    super.downloadFailed(error);
    this.source!.chunkManager.scheduleUpdateChunkPriorities();
  }

  toString() {
    return this.objectId.toString();
  }
}

/**
 * Chunk that contains the mesh for a single fragment of a single object.
 */
export class FragmentChunk extends Chunk {
  manifestChunk: ManifestChunk|null = null;
  fragmentId: FragmentId|null = null;
  vertexPositions: Float32Array|null = null;
  indices: Uint32Array|null = null;
  vertexNormals: Float32Array|null = null;
  constructor() {
    super();
  }
  initializeFragmentChunk(key: string, manifestChunk: ManifestChunk, fragmentId: FragmentId) {
    super.initialize(key);
    this.manifestChunk = manifestChunk;
    this.fragmentId = fragmentId;
  }
  freeSystemMemory() {
    this.manifestChunk = null;
    this.vertexPositions = this.indices = this.vertexNormals = null;
    this.fragmentId = null;
  }
  serialize(msg: any, transfers: any[]) {
    super.serialize(msg, transfers);
    msg['objectKey'] = this.manifestChunk!.key;
    let {vertexPositions, indices, vertexNormals} = this;
    msg['vertexPositions'] = vertexPositions;
    msg['indices'] = indices;
    msg['vertexNormals'] = vertexNormals;
    let vertexPositionsBuffer = vertexPositions!.buffer;
    transfers.push(vertexPositionsBuffer);
    let indicesBuffer = indices!.buffer;
    if (indicesBuffer !== vertexPositionsBuffer) {
      transfers.push(indicesBuffer);
    }
    let vertexNormalsBuffer = vertexNormals!.buffer;
    if (vertexNormalsBuffer !== vertexPositionsBuffer && vertexNormalsBuffer !== indicesBuffer) {
      transfers.push(vertexNormalsBuffer);
    }
    this.vertexPositions = this.indices = this.vertexNormals = null;
  }
  downloadSucceeded() {
    let {vertexPositions, indices, vertexNormals} = this;
    this.systemMemoryBytes = this.gpuMemoryBytes =
        vertexPositions!.byteLength + indices!.byteLength + vertexNormals!.byteLength;
    super.downloadSucceeded();
  }
}

/**
 * Assigns chunk.fragmentKeys to response[keysPropertyName].
 *
 * Verifies that response[keysPropertyName] is an array of strings.
 */
export function decodeJsonManifestChunk(
    chunk: ManifestChunk, response: any, keysPropertyName: string) {
  verifyObject(response);
  chunk.fragmentIds = verifyObjectProperty(response, keysPropertyName, verifyStringArray);
}

/**
 * Computes normal vectors for each vertex of a triangular mesh.
 *
 * The normal vector for each triangle with vertices (v0, v1, v2) is computed as the (normalized)
 * cross product of (v1 - v0, v2 - v1).  The normal vector for each vertex is obtained by averaging
 * the normal vector of each of the triangles that contains it.
 *
 * @param positions The vertex positions in [x0, y0, z0, x1, y1, z1, ...] format.
 * @param indices The indices of the triangle vertices.  Each triplet of consecutive values
 *     specifies a triangle.
 */
export function computeVertexNormals(positions: Float32Array, indices: Uint32Array) {
  const faceNormal = vec3.create();
  const v1v0 = vec3.create();
  const v2v1 = vec3.create();
  let vertexNormals = new Float32Array(positions.length);
  let numIndices = indices.length;
  for (let i = 0; i < numIndices; i += 3) {
    let i0 = indices[i] * 3, i1 = indices[i + 1] * 3, i2 = indices[i + 2] * 3;
    for (let j = 0; j < 3; ++j) {
      v1v0[j] = positions[i1 + j] - positions[i0 + j];
      v2v1[j] = positions[i2 + j] - positions[i1 + j];
    }
    vec3.cross(faceNormal, v1v0, v2v1);
    vec3.normalize(faceNormal, faceNormal);

    for (let k = 0; k < 3; ++k) {
      let index = indices[i + k];
      let offset = index * 3;
      for (let j = 0; j < 3; ++j) {
        vertexNormals[offset + j] += faceNormal[j];
      }
    }
  }
  // Normalize all vertex normals.
  let numVertices = vertexNormals.length;
  for (let i = 0; i < numVertices; i += 3) {
    let vec = <vec3>vertexNormals.subarray(i, i + 3);
    vec3.normalize(vec, vec);
  }
  return vertexNormals;
}

/**
 * Extracts vertex positions and indices of the specified endianness from `data'.
 *
 * The vertexByteOffset specifies the byte offset into `data' of the start of the vertex position
 * data.  The vertex data must consist of verticesPerPrimitive * numVertices 32-bit float values.
 *
 * If indexByteOffset is not specified, it defaults to the end of the vertex position data.  If
 * numPrimitives is not specified, it is assumed that the index data continues until the end of the
 * array.
 */
export function decodeVertexPositionsAndIndices(
    chunk: {vertexPositions: Float32Array | null, indices: Uint32Array | null},
    verticesPerPrimitive: number, data: ArrayBuffer, endianness: Endianness,
    vertexByteOffset: number, numVertices: number, indexByteOffset?: number,
    numPrimitives?: number) {
  let vertexPositions = new Float32Array(data, vertexByteOffset, numVertices * 3);
  convertEndian32(vertexPositions, endianness);

  if (indexByteOffset === undefined) {
    indexByteOffset = vertexByteOffset + 12 * numVertices;
  }

  let numIndices: number|undefined;
  if (numPrimitives !== undefined) {
    numIndices = numPrimitives * verticesPerPrimitive;
  }

  // For compatibility with Firefox, length argument must not be undefined.
  let indices = numIndices === undefined ? new Uint32Array(data, indexByteOffset) :
                                           new Uint32Array(data, indexByteOffset, numIndices);
  if (indices.length % verticesPerPrimitive !== 0) {
    throw new Error(
        `Number of indices is not a multiple of ${verticesPerPrimitive}: ${indices.length}.`);
  }
  convertEndian32(indices, endianness);

  chunk.vertexPositions = vertexPositions;
  chunk.indices = indices;
}

/**
 * Extracts vertex positions and triangle vertex indices of the specified endianness from `data'.
 *
 * Vertex normals are computed.
 *
 * See decodeVertexPositionsAndIndices above.
 */
export function decodeTriangleVertexPositionsAndIndices(
    chunk: FragmentChunk, data: ArrayBuffer, endianness: Endianness, vertexByteOffset: number,
    numVertices: number, indexByteOffset?: number, numTriangles?: number) {
  decodeVertexPositionsAndIndices(
      chunk, /*verticesPerPrimitive=*/3, data, endianness, vertexByteOffset, numVertices,
      indexByteOffset, numTriangles);
  chunk.vertexNormals = computeVertexNormals(chunk.vertexPositions!, chunk.indices!);
}

export interface MeshSource {
  // TODO(jbms): Move this declaration to class definition below and declare abstract once
  // TypeScript supports mixins with abstract classes.
  downloadFragment(chunk: FragmentChunk, cancellationToken: CancellationToken): Promise<void>;
}

export class MeshSource extends ChunkSource {
  fragmentSource: FragmentSource;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    let fragmentSource = this.fragmentSource =
        this.registerDisposer(rpc.getRef<FragmentSource>(options['fragmentSource']));
    fragmentSource.meshSource = this;
  }

  getChunk(objectId: Uint64, clipBounds?: Bounds) {
    const key = getObjectKey(objectId, clipBounds);
    let chunk = <ManifestChunk>this.chunks.get(key);
    if (chunk === undefined) {
      chunk = this.getNewChunk_(ManifestChunk);
      chunk.initializeManifestChunk(key, objectId, clipBounds);
      this.addChunk(chunk);
    }
    return chunk;
  }

  getFragmentChunk(manifestChunk: ManifestChunk, fragmentId: FragmentId) {
    // TODO(blakely): This ends up storing two copies of the fragment if the manifestChunk's key was
    // generated with a clipBounds. Ideally we'd key the fragments by "objectId/fragmentId" since it
    // doesn't matter what manifest chunk it was requested from, but we can't at the moment since
    // the frontend's fragmentSource.chunks is only updated when the chunkManager detects a chunk
    // has been updated. This results in the "inverse" fragments showing up, i.e. going from
    // clipBounds=>none shows all the fragments that were not contained within the starting clipping
    // bounds.
    //
    // let bareKey = getObjectKey(manifestChunk.objectId);
    // let key = `${bareKey}/${fragmentId}`;
    let key = `${manifestChunk.key}/${fragmentId}`;
    let fragmentSource = this.fragmentSource;
    let chunk = <FragmentChunk>fragmentSource.chunks.get(key);
    if (chunk === undefined) {
      chunk = fragmentSource.getNewChunk_(FragmentChunk);
      chunk.initializeFragmentChunk(key, manifestChunk, fragmentId);
      fragmentSource.addChunk(chunk);
    }
    return chunk;
  }
}

@registerSharedObject(FRAGMENT_SOURCE_RPC_ID)
export class FragmentSource extends ChunkSource {
  meshSource: MeshSource|null = null;
  download(chunk: FragmentChunk, cancellationToken: CancellationToken) {
    return this.meshSource!.downloadFragment(chunk, cancellationToken);
  }
}

@registerSharedObject(MESH_LAYER_RPC_ID)
export class MeshLayer extends SegmentationLayerSharedObjectCounterpart {
  source: MeshSource;
  chunkedGraph: ChunkedGraphLayer|null;
  private requestedChildChunks: Map<string, { add: Uint64[], delete: Uint64[] }>;

  private debouncedHandleChildChunks = debounce(() => {
    this.handleChildChunks();
  }, 100);

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.source = this.registerDisposer(rpc.getRef<MeshSource>(options['source']));
    this.chunkedGraph = this.registerDisposer(rpc.get(options['chunkedGraph']));
    this.registerDisposer(this.chunkManager.recomputeChunkPriorities.add(() => {
      this.updateChunkPriorities();
    }));

    this.requestedChildChunks = new Map<string, { add: Uint64[], delete: Uint64[] }>();
  }

  private handleChildChunks() {
    for (const [rootId, elements] of this.requestedChildChunks.entries()) {
      let rootTmp = Uint64.parseString(rootId);
      if (!this.rootSegments.has(rootTmp)) {
        console.log('Adding 3D children aborted due to missing root.');
        continue;
      }

      for (let e of elements.add) {
        this.visibleSegments3D.add(e);
        this.segmentEquivalences.link(rootTmp, e);
      }
      for (let e of elements.delete) {
        this.visibleSegments3D.delete(e);
      }
    }
    this.requestedChildChunks.clear();
  }

  private updateChunkPriorities() {
    const visibility = this.visibility.value;
    if (visibility === Number.NEGATIVE_INFINITY) {
      return;
    }
    const priorityTier = getPriorityTier(visibility);
    const basePriority = getBasePriority(visibility);
    const {source, chunkManager} = this;
    forEachVisibleSegment3D(this, (objectId, rootObjectId) => {
      let manifestChunk = source.getChunk(objectId, this.clipBounds.value);
      chunkManager.requestChunk(
          manifestChunk, priorityTier, basePriority + MESH_OBJECT_MANIFEST_CHUNK_PRIORITY);
      const state = manifestChunk.state;
      switch(state) {
        case ChunkState.SYSTEM_MEMORY_WORKER:
        case ChunkState.SYSTEM_MEMORY:
        case ChunkState.GPU_MEMORY: {
          for (let fragmentId of manifestChunk.fragmentIds!) {
            let fragmentChunk = source.getFragmentChunk(manifestChunk, fragmentId);
            chunkManager.requestChunk(
                fragmentChunk, priorityTier, basePriority + MESH_OBJECT_FRAGMENT_CHUNK_PRIORITY);
          }
          break;
        }
        case ChunkState.FAILED: {
          if (this.chunkedGraph === null) {
            break;
          }
          manifestChunk.state = ChunkState.REQUESTING_CHILDREN;
          let segmentID = objectId.clone();
          let rootID = rootObjectId.clone();
          this.chunkedGraph.getChildren(objectId).then(children => {
            if (DEBUG) { // with open 3D view, quickly select/deselect some segments
              if (segmentID.low !== objectId.low || segmentID.high !== objectId.high) {
                console.log(`SegmentID ${segmentID.toString()} does not match ObjectID ${objectId.toString()}`);
              }
              if (rootID.low !== rootObjectId.low || rootID.high !== rootObjectId.high) {
                console.log(`RootID ${rootID.toString()} does not match RootObjectID ${rootObjectId.toString()}`);
              }
            }

            manifestChunk.state = ChunkState.FAILED;
            if (!this.rootSegments.has(rootID)) {
              console.log('Adding 3D chunks aborted due to missing root object.');
              return;
            }
            if (!this.requestedChildChunks.has(rootID.toString())) {
              this.requestedChildChunks.set(rootID.toString(), { add: new Array<Uint64>(), delete: new Array<Uint64>() });
            }
            this.requestedChildChunks.get(rootID.toString())!.add.push(...children);
            this.requestedChildChunks.get(rootID.toString())!.delete.push(segmentID);

            this.debouncedHandleChildChunks();
          });
          break;
        }
      }
    });
  }
}
