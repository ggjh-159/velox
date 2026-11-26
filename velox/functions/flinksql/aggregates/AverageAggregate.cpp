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
#include "velox/functions/flinksql/aggregates/AverageAggregate.h"

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/experimental/stateful/agg/RetractableAggregate.h"
#include "velox/functions/lib/aggregates/AverageAggregateBase.h"
#include "velox/functions/lib/aggregates/DecimalAggregate.h"
#include "velox/type/DecimalUtil.h"

using namespace facebook::velox::functions::aggregate;

namespace facebook::velox::aggregate::flinksql {
namespace {

/// Average with retraction support built in, registered under the SQL name
/// so accumulate-only and retracting plans resolve to the same function.
/// Semantics follow Flink's AvgAggFunction: integral sums accumulate in
/// BIGINT with wrap-around addition and subtraction (uint64_t produces the
/// same bit pattern as Java's long arithmetic without signed-overflow UB),
/// integral results use truncating division, REAL results round the double
/// quotient, and the final value depends on the count alone - a count of
/// zero is an empty aggregate and reads as null however it was reached
/// (accumulation never saw input, or retraction took every row back).
/// Retraction subtracts the value from the sum and one from the count and
/// does not stop at zero: an over-retracted group keeps its negative count,
/// matching Flink, which does not guard the running values.
template <typename TInput, typename TResult, typename TAccumulator = uint64_t>
class AverageAggregate
    : public AverageAggregateBase<TInput, TAccumulator, TResult>,
      public stateful::RetractableAggregate {
 public:
  using Base = AverageAggregateBase<TInput, TAccumulator, TResult>;

  explicit AverageAggregate(TypePtr resultType) : Base(std::move(resultType)) {}

  // The base's extractAccumulators casts the intermediate sum column to
  // FlatVector<TAccumulator>, but a BIGINT column is a FlatVector<int64_t>,
  // which dynamic_cast cannot see as FlatVector<uint64_t>. The uint64_t sum
  // and the BIGINT column share the bit pattern, so extract through
  // int64_t and cast bit-preservingly.
  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    if constexpr (std::is_same_v<TAccumulator, uint64_t>) {
      auto rowVector = (*result)->as<RowVector>();
      auto sumVector = rowVector->childAt(0)->asFlatVector<int64_t>();
      auto countVector = rowVector->childAt(1)->asFlatVector<int64_t>();

      rowVector->resize(numGroups);
      sumVector->resize(numGroups);
      countVector->resize(numGroups);
      uint64_t* rawNulls = this->getRawNulls(rowVector);

      int64_t* rawCounts = countVector->mutableRawValues();
      int64_t* rawSums = sumVector->mutableRawValues();
      for (auto i = 0; i < numGroups; ++i) {
        char* group = groups[i];
        if (this->isNull(group)) {
          rowVector->setNull(i, true);
        } else {
          this->clearNull(rawNulls, i);
          auto* sumCount = this->accumulator(group);
          rawCounts[i] = sumCount->count;
          rawSums[i] = static_cast<int64_t>(sumCount->sum);
        }
      }
    } else {
      Base::extractAccumulators(groups, numGroups, result);
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<TResult>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = this->getRawNulls(vector);

    TResult* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (this->isNull(group)) {
        vector->setNull(i, true);
      } else {
        auto* sumCount = this->accumulator(group);
        if (sumCount->count == 0) {
          vector->setNull(i, true);
        } else {
          this->clearNull(rawNulls, i);
          if constexpr (std::is_integral_v<TResult>) {
            rawValues[i] = static_cast<TResult>(
                static_cast<int64_t>(sumCount->sum) / sumCount->count);
          } else {
            rawValues[i] =
                static_cast<TResult>(sumCount->sum / sumCount->count);
          }
        }
      }
    }
  }

  // Retraction mirrors the base's non-null accumulation: subtract the
  // argument from the sum and one from the count. The null filter matches
  // accumulation (null arguments never contributed), unsigned sums wrap
  // around like Java's long subtraction, and the count may go negative on
  // over-retraction.
  void retractRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    Base::decodedRaw_.decode(*args[0], rows);
    rows.applyToSelected([&](vector_size_t i) {
      if (Base::decodedRaw_.isNullAt(i)) {
        return;
      }
      auto* sumCount = Base::accumulator(groups[i]);
      sumCount->sum -= static_cast<TAccumulator>(
          Base::decodedRaw_.template valueAt<TInput>(i));
      sumCount->count -= 1;
    });
  }
};

/// Average over DECIMAL(p, s) with Flink semantics: the sum stays at the
/// input scale while the result type is DECIMAL(38, max(6, s)), so the final
/// division rescales the fraction by 10^k, k = max(6, s) - s, with HALF_UP
/// rounding, matching Flink's decimal division. Retraction runs the negated
/// input through the same carry-tracking addition the accumulator uses: the
/// stored pair (overflow, sum) represents overflow * 2^127 + sum, so a
/// subtract carries or borrows through the overflow word symmetric to an
/// add, and the value follows the count alone like the fixed-width average.
template <typename TInput>
class DecimalAverageAggregate : public DecimalAggregate<int128_t, TInput>,
                                public stateful::RetractableAggregate {
 public:
  using Base = DecimalAggregate<int128_t, TInput>;
  using Acc = functions::aggregate::LongDecimalWithOverflowState;

  explicit DecimalAverageAggregate(TypePtr resultType, uint8_t resultRescale)
      : Base(std::move(resultType)), resultRescale_(resultRescale) {}

  // The base loop reads a final value for every non-null group, but a count
  // of zero is an empty aggregate and reads as null however it was reached,
  // so the zero-count check lives here; computeFinalValue cannot express
  // null.
  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<int128_t>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = this->getRawNulls(vector);

    int128_t* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (this->isNull(group)) {
        vector->setNull(i, true);
      } else {
        auto* sumCount = exec::Aggregate::value<Acc>(group);
        if (sumCount->count == 0) {
          vector->setNull(i, true);
        } else {
          this->clearNull(rawNulls, i);
          rawValues[i] = computeFinalValue(sumCount);
        }
      }
    }
  }

  // 'accumulator->count' is never zero: extractValues reads null for a
  // zero count. A negative count (over-retraction) flows through the
  // divisions, reading back the retracted input like the fixed-width
  // average.
  int128_t computeFinalValue(Acc* accumulator) override {
    const int128_t sum = accumulator->sum;
    const int64_t count = accumulator->count;

    // Decompose trueSum = overflow * 2^127 + sum (see
    // DecimalUtil::addWithOverflow) into quotient and remainder over count.
    // With no carry the native 128-bit division is exact; with a carry the
    // 2^127 part is split separately, mirroring DecimalUtil::computeAverage,
    // so neither division overflows 128 bits.
    int128_t quotient;
    int128_t remainder;
    if (accumulator->overflow == 0) {
      quotient = sum / count;
      remainder = sum % count;
    } else {
      uint128_t quotientMul;
      const uint128_t remainderMul =
          DecimalUtil::divideWithRoundUp<uint128_t, uint128_t, int64_t>(
              quotientMul, DecimalUtil::kOverflowMultiplier, count, true, 0, 0);
      int128_t quotientSum;
      const int128_t remainderSum =
          DecimalUtil::divideWithRoundUp<int128_t, int128_t, int64_t>(
              quotientSum, sum, count, true, 0, 0);
      quotient =
          checkedMultiply<int128_t>(
              static_cast<int128_t>(quotientMul), accumulator->overflow) +
          quotientSum;
      remainder =
          checkedMultiply<int128_t>(
              static_cast<int128_t>(remainderMul), accumulator->overflow) +
          remainderSum;
    }

    // trueSum / count = (quotient + extraQuotient) + fractionRemainder /
    // count. Rescale to the result scale with HALF_UP rounding on the
    // fraction: result = (quotient + extraQuotient) * 10^k +
    // round(fractionRemainder * 10^k / count). The dividend of the second
    // division is bounded by count, so rescaling it never overflows for the
    // k = max(6, s) - s <= 6 produced by Flink's avg result type.
    const int128_t extraQuotient = remainder / count;
    int128_t scaledRemainder;
    DecimalUtil::divideWithRoundUp<int128_t, int128_t, int64_t>(
        scaledRemainder, remainder % count, count, false, resultRescale_, 0);
    return checkedMultiply<int128_t>(
               quotient + extraQuotient,
               DecimalUtil::kPowersOfTen[resultRescale_]) +
        scaledRemainder;
  }

  // Retraction mirrors updateNonNullValue: the negated input runs through
  // the same addWithOverflow, whose carry lands in the overflow word in
  // either direction (a borrow is a carry of -1). DECIMAL(38) values fit
  // the 127-bit magnitude, so the negation is always exact. The null filter
  // matches accumulation, and the count may go negative on over-retraction.
  void retractRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    DecodedVector decoded;
    decoded.decode(*args[0], rows);
    rows.applyToSelected([&](vector_size_t i) {
      if (decoded.isNullAt(i)) {
        return;
      }
      auto* accumulator = exec::Aggregate::value<Acc>(groups[i]);
      const int128_t negated =
          -static_cast<int128_t>(decoded.valueAt<TInput>(i));
      accumulator->overflow += DecimalUtil::addWithOverflow(
          accumulator->sum, negated, accumulator->sum);
      accumulator->count -= 1;
    });
  }

 private:
  /// Extra fractional digits of the result type over the input type.
  uint8_t resultRescale_;
};

uint8_t decimalScaleOf(const TypePtr& type) {
  VELOX_CHECK(type->isDecimal());
  if (type->isShortDecimal()) {
    return static_cast<const ShortDecimalType&>(*type).scale();
  }
  return static_cast<const LongDecimalType&>(*type).scale();
}

} // namespace

/// Count is BIGINT. Following Flink's AvgAggFunction, the sum and result
/// types depend on the input type:
///       INPUT TYPE    |     SUM           |     AVG
///   ------------------|-------------------|------------------------
///     TINYINT         |     BIGINT        |    TINYINT
///     SMALLINT        |     BIGINT        |    SMALLINT
///     INTEGER         |     BIGINT        |    INTEGER
///     BIGINT          |     BIGINT        |    BIGINT
///     REAL            |     DOUBLE        |    REAL
///     DOUBLE          |     DOUBLE        |    DOUBLE
///     DECIMAL(p, s)   |     DECIMAL       |    DECIMAL(38, max(6, s))
/// The integral sums wrap around like Java's long, the integral results use
/// truncating division, and REAL results round the double quotient to
/// FLOAT, all matching Flink's AvgAggFunction.
void registerAverageAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  for (const auto& inputType : {"tinyint", "smallint", "integer", "bigint"}) {
    signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                             .returnType(inputType)
                             .intermediateType("row(bigint,bigint)")
                             .argumentType(inputType)
                             .build());
  }

  for (const auto& inputType : {"real", "double"}) {
    signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                             .returnType(inputType)
                             .intermediateType("row(double,bigint)")
                             .argumentType(inputType)
                             .build());
  }

  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .integerVariable("a_precision")
                           .integerVariable("a_scale")
                           .integerVariable("r_scale", "max(6, a_scale)")
                           .argumentType("DECIMAL(a_precision, a_scale)")
                           .intermediateType("varbinary")
                           .returnType("DECIMAL(38, r_scale)")
                           .build());

  const auto name = prefix + "avg";
  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        if (exec::isRawInput(step)) {
          VELOX_CHECK_EQ(
              argTypes.size(), 1, "{} takes exactly one argument", name);
          const auto& inputType = argTypes[0];
          switch (inputType->kind()) {
            case TypeKind::TINYINT:
              return std::make_unique<AverageAggregate<int8_t, int8_t>>(
                  resultType);
            case TypeKind::SMALLINT:
              return std::make_unique<AverageAggregate<int16_t, int16_t>>(
                  resultType);
            case TypeKind::INTEGER:
              return std::make_unique<AverageAggregate<int32_t, int32_t>>(
                  resultType);
            case TypeKind::BIGINT: {
              if (inputType->isShortDecimal()) {
                return std::make_unique<DecimalAverageAggregate<int64_t>>(
                    resultType,
                    decimalScaleOf(resultType) - decimalScaleOf(inputType));
              }
              return std::make_unique<AverageAggregate<int64_t, int64_t>>(
                  resultType);
            }
            case TypeKind::HUGEINT: {
              if (inputType->isLongDecimal()) {
                return std::make_unique<DecimalAverageAggregate<int128_t>>(
                    resultType,
                    decimalScaleOf(resultType) - decimalScaleOf(inputType));
              }
              VELOX_NYI();
            }
            case TypeKind::REAL:
              return std::make_unique<AverageAggregate<float, float, double>>(
                  resultType);
            case TypeKind::DOUBLE:
              return std::make_unique<AverageAggregate<double, double, double>>(
                  resultType);
            default:
              VELOX_FAIL(
                  "Unknown input type for {} aggregation {}",
                  name,
                  inputType->kindName());
          }
        } else {
          // TODO: Support partial-to-final aggregation steps.
          VELOX_USER_FAIL(
              "{} does not support partial-to-final aggregation steps yet",
              name);
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::flinksql
