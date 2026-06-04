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
 * @file Frontend chunk sources for the precomputed spatially-indexed skeleton
 * subsource ("show all skeletons in view"). The backend
 * (`PrecomputedSpatialSkeletonSourceBackend`) reads a `*.spatial` index file
 * for each chunk and assembles the listed skeleton fragments; here we only set
 * up the chunk grid and reuse the shared spatially-indexed skeleton rendering
 * machinery (the same path used by CATMAID and zarr-vectors).
 *
 * Coordinate frame: the spatial grid is expressed in nanometers, shifted so the
 * grid origin sits at 0 (the shader `spatialChunkCull` clips against
 * `chunkGridPosition * chunkLayout.size`, which assumes a phase-0 grid). The
 * backend packs `vertexNm - gridOriginNm`, and the caller supplies a
 * `subsourceToModelSubspaceTransform` that maps this shifted-nm frame into the
 * volume's model space.
 */

import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { WithParameters } from "#src/chunk_manager/frontend.js";
import { PrecomputedSpatialSkeletonSourceParameters } from "#src/datasource/precomputed/base.js";
import { WithSharedKvStoreContext } from "#src/kvstore/chunk_source_frontend.js";
import type { SharedKvStoreContext } from "#src/kvstore/frontend.js";
import type { VertexAttributeInfo } from "#src/skeleton/base.js";
import {
  MultiscaleSpatiallyIndexedSkeletonSource,
  SPATIAL_SKELETON_SOURCE_OPTIONS,
  SpatiallyIndexedSkeletonSource,
  type SpatiallyIndexedSkeletonChunkSpecification,
} from "#src/skeleton/frontend.js";
import type { SliceViewSourceOptions } from "#src/sliceview/base.js";
import { makeSliceViewChunkSpecification } from "#src/sliceview/base.js";
import { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import type { SliceViewSingleResolutionSource } from "#src/sliceview/frontend.js";
import { DataType } from "#src/util/data_type.js";
import { mat4, vec3 } from "#src/util/geom.js";
import { getShaderType } from "#src/webgl/shader_lib.js";
import {
  computeTextureFormat,
  TextureFormat,
} from "#src/webgl/texture_access.js";

// 1 nanometer expressed in meters; chunk coordinates are in nm and the render
// layer's display space is in meters, so the chunk layout scales nm -> m
// (matching the CATMAID spatial skeleton source).
const NANOMETERS_TO_METERS = 1e-9;

// Shape the spatially-indexed skeleton render layer reads off each vertex
// attribute (matches the non-exported `VertexAttributeRenderInfo` in
// skeleton/frontend.ts).
interface SpatialVertexAttributeRenderInfo {
  name: string;
  dataType: DataType;
  numComponents: number;
  webglDataType: number;
  glslDataType: string;
}

function webglDataTypeFor(dataType: DataType): number {
  switch (dataType) {
    case DataType.FLOAT32:
      return WebGL2RenderingContext.FLOAT;
    case DataType.UINT8:
      return WebGL2RenderingContext.UNSIGNED_BYTE;
    case DataType.INT8:
      return WebGL2RenderingContext.BYTE;
    case DataType.UINT16:
      return WebGL2RenderingContext.UNSIGNED_SHORT;
    case DataType.INT16:
      return WebGL2RenderingContext.SHORT;
    case DataType.UINT32:
    case DataType.UINT64:
      return WebGL2RenderingContext.UNSIGNED_INT;
    case DataType.INT32:
      return WebGL2RenderingContext.INT;
    default:
      throw new Error(`Unsupported skeleton attribute DataType: ${dataType}`);
  }
}

// Builds the render-layer attribute list: position (slot 0, implicit), the
// uint64 `segment` id (slot 1, must match the base source so picking/coloring
// work), then the info file's per-vertex attributes (e.g. radius,
// cross_sectional_area) so they are uploaded and exposed as `prop_<name>()`.
// Order matches what the backend packs into `chunk.vertexAttributes`.
function buildPrecomputedSpatialVertexAttributes(
  vertexAttributes: Map<string, VertexAttributeInfo>,
): SpatialVertexAttributeRenderInfo[] {
  const out: SpatialVertexAttributeRenderInfo[] = [
    {
      name: "",
      dataType: DataType.FLOAT32,
      numComponents: 3,
      webglDataType: WebGL2RenderingContext.FLOAT,
      glslDataType: "vec3",
    },
    {
      name: "segment",
      dataType: DataType.UINT64,
      numComponents: 1,
      webglDataType: WebGL2RenderingContext.UNSIGNED_INT,
      glslDataType: getShaderType(DataType.UINT64, 1),
    },
    // Synthesized per-vertex tangent (unit average of incident edge
    // directions), packed by the backend. Drives `prop_tangent()` for
    // directional coloring of both edges and nodes.
    {
      name: "tangent",
      dataType: DataType.FLOAT32,
      numComponents: 3,
      webglDataType: WebGL2RenderingContext.FLOAT,
      glslDataType: "vec3",
    },
  ];
  for (const [name, info] of vertexAttributes) {
    out.push({
      name,
      dataType: info.dataType,
      numComponents: info.numComponents,
      webglDataType: webglDataTypeFor(info.dataType),
      glslDataType: getShaderType(info.dataType, info.numComponents),
    });
  }
  return out;
}

export class PrecomputedSpatiallyIndexedSkeletonSource extends WithParameters(
  WithSharedKvStoreContext(SpatiallyIndexedSkeletonSource),
  PrecomputedSpatialSkeletonSourceParameters,
) {
  private precomputedAttributeTextureFormats_?: TextureFormat[];

  constructor(
    ...args: ConstructorParameters<typeof SpatiallyIndexedSkeletonSource>
  ) {
    super(...args);
    // Replace the base `[position, segment]` with one that also exposes the
    // info file's per-vertex attributes (matches the backend's packing).
    this.vertexAttributes = buildPrecomputedSpatialVertexAttributes(
      this.parameters.metadata.vertexAttributes,
    );
  }

  get attributeTextureFormats(): TextureFormat[] {
    let cached = this.precomputedAttributeTextureFormats_;
    if (cached === undefined) {
      cached = this.precomputedAttributeTextureFormats_ =
        this.vertexAttributes.map(({ dataType, numComponents }) =>
          computeTextureFormat(new TextureFormat(), dataType, numComponents),
        );
    }
    return cached;
  }
}

export class PrecomputedMultiscaleSpatiallyIndexedSkeletonSource extends MultiscaleSpatiallyIndexedSkeletonSource {
  private readonly chunkSizeNm: Float32Array;
  private readonly extentNm: Float32Array;
  private readonly parameters: PrecomputedSpatialSkeletonSourceParameters;

  constructor(
    chunkManager: ChunkManager,
    private sharedKvStoreContext: SharedKvStoreContext,
    options: {
      parameters: PrecomputedSpatialSkeletonSourceParameters;
      // Chunk size, in nanometers.
      chunkSizeNm: Float32Array;
      // Total extent of the grid in nanometers (upper bound in the shifted,
      // gridOrigin-relative frame; the lower bound is 0).
      extentNm: Float32Array;
    },
  ) {
    super(chunkManager);
    this.parameters = options.parameters;
    this.chunkSizeNm = options.chunkSizeNm;
    this.extentNm = options.extentNm;
  }

  get rank(): number {
    return 3;
  }

  getSpatialSkeletonGridSizes(): { x: number; y: number; z: number }[] {
    return [
      {
        x: this.chunkSizeNm[0],
        y: this.chunkSizeNm[1],
        z: this.chunkSizeNm[2],
      },
    ];
  }

  getPerspectiveSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    const sources = this.getSources(SPATIAL_SKELETON_SOURCE_OPTIONS);
    return sources.length > 0 ? sources[0] : [];
  }

  getSliceViewPanelSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    return this.getPerspectiveSources();
  }

  getSources(
    _options: SliceViewSourceOptions,
  ): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[][] {
    const chunkDataSize = Uint32Array.from([
      this.chunkSizeNm[0],
      this.chunkSizeNm[1],
      this.chunkSizeNm[2],
    ]);

    // Chunk grid coordinates are in nm; convert to meters for the render
    // layer's display space (identical to CATMAID).
    const chunkLayoutTransform = mat4.create();
    mat4.fromScaling(
      chunkLayoutTransform,
      vec3.fromValues(
        NANOMETERS_TO_METERS,
        NANOMETERS_TO_METERS,
        NANOMETERS_TO_METERS,
      ),
    );
    const chunkLayout = new ChunkLayout(
      vec3.fromValues(chunkDataSize[0], chunkDataSize[1], chunkDataSize[2]),
      chunkLayoutTransform,
      3,
    );

    const spec: SpatiallyIndexedSkeletonChunkSpecification = {
      ...makeSliceViewChunkSpecification({
        rank: 3,
        chunkDataSize,
        lowerVoxelBound: new Float32Array(3),
        upperVoxelBound: this.extentNm,
      }),
      chunkLayout,
    };

    const chunkSource = this.chunkManager.getChunkSource(
      PrecomputedSpatiallyIndexedSkeletonSource,
      {
        sharedKvStoreContext: this.sharedKvStoreContext,
        spec,
        parameters: this.parameters,
      },
    );

    return [[{ chunkSource, chunkToMultiscaleTransform: mat4.create() }]];
  }
}
