/**
 * @license
 * Copyright 2017 The Neuroglancer Authors
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

import {ChunkManager} from 'neuroglancer/chunk_manager/frontend';
import {CHUNKED_GRAPH_LAYER_RPC_ID} from 'neuroglancer/chunked_graph/base';
import {SegmentSelection, SegmentationDisplayState} from 'neuroglancer/segmentation_display_state/frontend';
import {DataType} from 'neuroglancer/sliceview/base';
import {MultiscaleSliceViewChunkSource} from 'neuroglancer/sliceview/frontend';
import {RenderLayer as GenericSliceViewRenderLayer} from 'neuroglancer/sliceview/renderlayer';
import {Uint64Set} from 'neuroglancer/uint64_set';
import {openHttpRequest, sendHttpJsonPostRequest, HttpError} from 'neuroglancer/util/http_request';
import {Uint64} from 'neuroglancer/util/uint64';
import {RPC} from 'neuroglancer/worker_rpc';
import {VolumeChunkSource as VolumeChunkSourceInterface, VolumeChunkSpecification, VolumeSourceOptions} from 'neuroglancer/sliceview/volume/base';
import {VolumeChunkSource} from 'neuroglancer/sliceview/volume/frontend';

export const GRAPH_SERVER_NOT_SPECIFIED = Symbol('Graph Server Not Specified.');

export interface SegmentSelection {
  segment: Uint64;
  root: Uint64;
  position: number[];
}

export class ChunkedGraphChunkSource extends VolumeChunkSource implements
    VolumeChunkSourceInterface {
  rootSegments: Uint64Set;

  constructor(chunkManager: ChunkManager, options: {
      spec: VolumeChunkSpecification, rootSegments: Uint64Set}) {
    super(chunkManager, options);
    this.rootSegments = options.rootSegments;
  }

  initializeCounterpart(rpc: RPC, options: any) {
    options['rootSegments'] = this.rootSegments.rpcId;
    super.initializeCounterpart(rpc, options);
  }
}

export interface MultiscaleChunkedGraphSource extends MultiscaleSliceViewChunkSource {
  getSources: (options: VolumeSourceOptions) => VolumeChunkSource[][];

  dataType: DataType;
}

export class ChunkedGraphLayer extends GenericSliceViewRenderLayer {
  private graphurl: string;

  constructor(
      chunkManager: ChunkManager,
      url: string,
      public sources: ChunkedGraphChunkSource[][],
      displayState: SegmentationDisplayState) {
    super(chunkManager, sources, {
      rpcTransfer: {
        'chunkManager': chunkManager.rpcId,
        'url': url,
        'rootSegments': displayState.rootSegments.rpcId,
        'visibleSegments3D': displayState.visibleSegments3D.rpcId,
        'segmentEquivalences': displayState.segmentEquivalences.rpcId
      },
      rpcType: CHUNKED_GRAPH_LAYER_RPC_ID
    });
    this.graphurl = url;
  }

  get url() {
    return this.graphurl;
  }

  getRoot(selection: SegmentSelection): Promise<Uint64> {
    const {url} = this;
    if (url === '') {
      return Promise.resolve(selection.segmentId);
    }

    let promise = sendHttpJsonPostRequest(openHttpRequest(`${url}/1.0/graph/root`, 'POST'),
        [String(selection.segmentId), ...selection.position],
        'arraybuffer');

    return promise.then(response => {
      let uint32 = new Uint32Array(response);
      return new Uint64(uint32[0], uint32[1]);
    }).catch((e: HttpError) => {
      console.log(`Could not retrieve root for segment ${selection.segmentId}`);
      console.error(e);
      return Promise.reject(e);
    });
  }

  mergeSegments(first: SegmentSelection, second: SegmentSelection): Promise<Uint64> {
    const {url} = this;
    if (url === '') {
      return Promise.reject(GRAPH_SERVER_NOT_SPECIFIED);
    }

    let promise = sendHttpJsonPostRequest(openHttpRequest(`${url}/1.0/graph/merge`, 'POST'),
      [
        [String(first.segmentId), ...first.position], [String(second.segmentId), ...second.position]
      ],
      'arraybuffer');

    return promise.then(response => {
      let uint32 = new Uint32Array(response);
      return new Uint64(uint32[0], uint32[1]);
    }).catch((e: HttpError) => {
      console.log(`Could not retrieve merge result of segments ${first.segmentId} and ${second.segmentId}.`);
      console.error(e);
      return Promise.reject(e);
    });
  }

  splitSegments(first: SegmentSelection[], second: SegmentSelection[]): Promise<Uint64[]> {
    const {url} = this;
    if (url === '') {
      return Promise.reject(GRAPH_SERVER_NOT_SPECIFIED);
    }

    let promise = sendHttpJsonPostRequest(openHttpRequest(`${url}/1.0/graph/split`, 'POST'),
      {
        'sources': first.map(x => [String(x.segmentId), ...x.position]),
        'sinks': second.map(x => [String(x.segmentId), ...x.position])
      },
      'arraybuffer');

    return promise.then(response => {
      let uint32 = new Uint32Array(response);
      let final: Uint64[] = new Array(uint32.length / 2);
      for (let i = 0; i < uint32.length / 2; i++) {
        final[i] = new Uint64(uint32[2 * i], uint32[2 * i + 1]);
      }
      return final;
    }).catch((e: HttpError) => {
      console.log(`Could not retrieve split result.`);// of segments ${first} and ${second}.`);
      console.error(e);
      return Promise.reject(e);
    });
  }

  draw() {}
}
