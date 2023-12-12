/* Copyright 2022 The StableHLO Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_DIALECT_CHLO_BYTECODE_H
#define STABLEHLO_DIALECT_CHLO_BYTECODE_H

namespace mlir {
namespace chlo {
class ChloDialect;

// Add the interface necessary for encoding and decoding the CHLO dialect
// components in bytecode.
void addBytecodeInterface(ChloDialect *dialect);
}  // namespace chlo
}  // namespace mlir

#endif  // CHLO_DIALECT_CHLO_BYTECODE_H
