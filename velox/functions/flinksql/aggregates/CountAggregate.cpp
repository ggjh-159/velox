/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/functions/flinksql/aggregates/CountAggregate.h"

#include <vector>

#include "velox/experimental/stateful/agg/RetractableAggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/aggregates/SimpleNumericAggregate.h"

using namespace facebook::velox::functions::aggregate;

namespace facebook::velox::aggregate::flinksql {

namespace {

/// Count with retraction support built in, registered under the SQL name so
/// accumulate-only and retracting plans resolve to the same function.
/// Semantics follow Flink's CountAggFunction/Count1AggFunction: the no-arg
/// form (count(*)) counts every row, the one-arg form counts non-null
/// arguments; both filters apply to accumulation and retraction alike, and
/// retraction does not stop at zero - an over-retracted group keeps its
/// negative count, matching Flink, which does not guard the running sum.
class CountAggregate : public SimpleNumericAggregate<bool, int64_t, int64_t>,
                       public stateful::RetractableAggregate {
  using BaseAggregate = SimpleNumericAggregate<bool, int64_t, int64_t>;

 public:
  explicit CountAggregate() : BaseAggregate(BIGINT()) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(int64_t);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::doExtractValues(groups, numGroups, result, [&](char* group) {
      return *value<int64_t>(group);
    });
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    countSelected(groups, rows, args, 1);
  }

  void retractRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    countSelected(groups, rows, args, -1);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);
    rows.applyToSelected([&](vector_size_t i) {
      addToGroup(groups[i], decodedIntermediate_.valueAt<int64_t>(i));
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    if (args.empty()) {
      addToGroup(group, rows.countSelected());
      return;
    }

    DecodedVector decoded(*args[0], rows);
    if (decoded.isConstantMapping()) {
      if (!decoded.isNullAt(0)) {
        addToGroup(group, rows.countSelected());
      }
    } else if (decoded.mayHaveNulls()) {
      int64_t nonNullCount = 0;
      rows.applyToSelected([&](vector_size_t i) {
        if (!decoded.isNullAt(i)) {
          ++nonNullCount;
        }
      });
      addToGroup(group, nonNullCount);
    } else {
      addToGroup(group, rows.countSelected());
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    int64_t count = 0;
    if (decodedIntermediate_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedIntermediate_.isNullAt(i)) {
          count += decodedIntermediate_.valueAt<int64_t>(i);
        }
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        count += decodedIntermediate_.valueAt<int64_t>(i);
      });
    }

    addToGroup(group, count);
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto i : indices) {
      // result of count is never null
      *value<int64_t>(groups[i]) = static_cast<int64_t>(0);
    }
  }

 private:
  // Adds 'delta' (+1 accumulation, -1 retraction) per selected row, with the
  // argument's null filter applying to both directions.
  void countSelected(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      int64_t delta) {
    if (args.empty()) {
      rows.applyToSelected(
          [&](vector_size_t i) { addToGroup(groups[i], delta); });
      return;
    }

    DecodedVector decoded(*args[0], rows);
    if (decoded.isConstantMapping()) {
      if (!decoded.isNullAt(0)) {
        rows.applyToSelected(
            [&](vector_size_t i) { addToGroup(groups[i], delta); });
      }
    } else if (decoded.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decoded.isNullAt(i)) {
          addToGroup(groups[i], delta);
        }
      });
    } else {
      rows.applyToSelected(
          [&](vector_size_t i) { addToGroup(groups[i], delta); });
    }
  }

  inline void addToGroup(char* group, int64_t count) {
    *value<int64_t>(group) += count;
  }

  DecodedVector decodedIntermediate_;
};

} // namespace

void registerCountAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("bigint")
          .intermediateType("bigint")
          .build(),
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("bigint")
          .intermediateType("bigint")
          .argumentType("T")
          .build(),
  };

  const auto name = prefix + "count";
  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& /*resultType*/,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        return std::make_unique<CountAggregate>();
      },
      {false /*orderSensitive*/, false /*companionFunction*/},
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::flinksql
