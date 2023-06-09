// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export * from 'onnxruntime-common';
import {registerBackend} from 'onnxruntime-common';
import {onnxruntimeBackend} from './backend';

registerBackend('cpu', onnxruntimeBackend, 100);
registerBackend('cuda', onnxruntimeBackend, 100);
registerBackend('directml', onnxruntimeBackend, 100);
registerBackend('coreml', onnxruntimeBackend, 100);
