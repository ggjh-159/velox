# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(velox_functions_flink_aggregates_test
                     AverageAggregateTest.cpp
                     CountAggregateTest.cpp
                     MinMaxRetractAccumulatorTest.cpp)

add_test(NAME velox_functions_flink_aggregates_test
         COMMAND velox_functions_flink_aggregates_test)

target_link_libraries(
  velox_functions_flink_aggregates_test
  velox_functions_flink_aggregates
  velox_core
  velox_exec
  velox_functions_lib
  velox_type
  velox_vector
  velox_vector_test_lib
  GTest::gtest
  GTest::gtest_main)
