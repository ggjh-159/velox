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
#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/RowContainer.h"
#include "velox/experimental/stateful/agg/RetractableAggregate.h"
#include "velox/functions/flinksql/aggregates/CountAggregate.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::aggregate::flinksql::test {
namespace {

class CountAggregateTest : public ::testing::Test,
                           public facebook::velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
    registerCountAggregate("", true, true);
  }

  // Drives one single-group accumulator directly - the way the group
  // aggregation operator drives aggregate functions outside the plan path -
  // so retraction can be exercised without a retract-aware plan.
  class Driver {
   public:
    Driver(std::vector<TypePtr> argTypes, memory::MemoryPool* pool)
        : fn_(exec::Aggregate::create(
              "count",
              core::AggregationNode::Step::kSingle,
              std::move(argTypes),
              BIGINT(),
              queryConfig())),
          allocator_(pool) {
      fn_->setAllocator(&allocator_);
      const int32_t rowSizeOffset = bits::nbytes(1);
      int32_t offset = rowSizeOffset + sizeof(uint32_t);
      offset = bits::roundUp(offset, fn_->accumulatorAlignmentSize());
      fn_->setOffsets(
          offset,
          exec::RowContainer::nullByte(0),
          exec::RowContainer::nullMask(0),
          exec::RowContainer::initializedByte(0),
          exec::RowContainer::initializedMask(0),
          rowSizeOffset);
      row_.resize(offset + fn_->accumulatorFixedWidthSize());
      std::vector<char*> groups{row_.data()};
      const std::vector<vector_size_t> indices{0};
      fn_->initializeNewGroups(groups.data(), indices);
    }

    // count(*): accumulates/retracts 'numRows' unconditional contributions.
    void addCountStar(int32_t numRows) {
      run(numRows, nullptr, /*retract=*/false);
    }

    void retractCountStar(int32_t numRows) {
      run(numRows, nullptr, /*retract=*/true);
    }

    // count(x): accumulates/retracts the non-null rows of 'arg'.
    void add(const VectorPtr& arg) {
      run(arg->size(), arg, /*retract=*/false);
    }

    void retract(const VectorPtr& arg) {
      run(arg->size(), arg, /*retract=*/true);
    }

    int64_t count() {
      auto result = BaseVector::create(BIGINT(), 1, poolOfFn());
      std::vector<char*> groups{row_.data()};
      fn_->extractValues(groups.data(), 1, &result);
      return result->as<FlatVector<int64_t>>()->valueAt(0);
    }

    VectorPtr extractIntermediate() {
      auto result = BaseVector::create(BIGINT(), 1, poolOfFn());
      std::vector<char*> groups{row_.data()};
      fn_->extractAccumulators(groups.data(), 1, &result);
      return result;
    }

    void addIntermediate(const VectorPtr& intermediate) {
      std::vector<char*> groups{row_.data()};
      fn_->addIntermediateResults(
          groups.data(), SelectivityVector(1), {intermediate}, false);
    }

   private:
    static core::QueryConfig queryConfig() {
      return core::QueryConfig({});
    }

    memory::MemoryPool* poolOfFn() {
      return allocator_.pool();
    }

    void run(int32_t numRows, const VectorPtr& arg, bool retract) {
      std::vector<VectorPtr> args;
      if (arg != nullptr) {
        args.push_back(arg);
      }
      const SelectivityVector rows(numRows);
      std::vector<char*> groups(numRows, row_.data());
      if (retract) {
        auto* retractable =
            dynamic_cast<stateful::RetractableAggregate*>(fn_.get());
        VELOX_CHECK_NOT_NULL(retractable);
        retractable->retractRawInput(groups.data(), rows, args, false);
      } else {
        fn_->addRawInput(groups.data(), rows, args, false);
      }
    }

    // The aggregate must outlive the allocator feeding it.
    std::unique_ptr<exec::Aggregate> fn_;
    HashStringAllocator allocator_;
    std::vector<char> row_;
  };
};

TEST_F(CountAggregateTest, oneArgRetractSkipsNulls) {
  Driver driver({BIGINT()}, pool());
  driver.add(makeNullableFlatVector<int64_t>({1, std::nullopt, 2}));
  EXPECT_EQ(2, driver.count());
  // Only the non-null argument retracts; the null row leaves the count alone.
  driver.retract(makeNullableFlatVector<int64_t>({std::nullopt, 3}));
  EXPECT_EQ(1, driver.count());
  driver.retract(makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}));
  EXPECT_EQ(1, driver.count());
}

TEST_F(CountAggregateTest, countStarRetractsUnconditionally) {
  Driver driver({}, pool());
  driver.addCountStar(3);
  EXPECT_EQ(3, driver.count());
  driver.retractCountStar(1);
  EXPECT_EQ(2, driver.count());
}

TEST_F(CountAggregateTest, overRetractKeepsNegativeCount) {
  Driver driver({}, pool());
  driver.addCountStar(1);
  driver.retractCountStar(2);
  // Retraction does not stop at zero: the group keeps the negative count and
  // the operator judges liveness by ==0 only, matching Flink.
  EXPECT_EQ(-1, driver.count());
}

TEST_F(CountAggregateTest, intermediateRoundTripCarriesNegativeCount) {
  Driver source({}, pool());
  source.addCountStar(1);
  source.retractCountStar(2);
  Driver restored({}, pool());
  restored.addIntermediate(source.extractIntermediate());
  EXPECT_EQ(-1, restored.count());
  restored.addCountStar(3);
  EXPECT_EQ(2, restored.count());
}

TEST_F(CountAggregateTest, plainNameCarriesRetractBothArities) {
  // Both arities resolve to the same registration, which carries the retract
  // capability - accumulate-only and retracting plans share one function.
  for (const auto& argTypes :
       {std::vector<TypePtr>{}, std::vector<TypePtr>{BIGINT()}}) {
    auto fn = exec::Aggregate::create(
        "count",
        core::AggregationNode::Step::kSingle,
        argTypes,
        BIGINT(),
        core::QueryConfig({}));
    EXPECT_NE(dynamic_cast<stateful::RetractableAggregate*>(fn.get()), nullptr);
  }
}

} // namespace
} // namespace facebook::velox::aggregate::flinksql::test
