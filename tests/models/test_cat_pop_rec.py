# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import pytest
from pyspark.sql import functions as sf

from replay.models import CatPopRec
from tests.utils import spark, sparkDataFrameEqual


@pytest.fixture
def cat_tree(spark):
    return spark.createDataFrame(
        data=[
            [None, "healthy_food"],
            [None, "groceries"],
            ["groceries", "fruits"],
            ["fruits", "apples"],
            ["fruits", "bananas"],
            ["apples", "red_apples"],
        ],
        schema="parent_cat string, category string",
    )


@pytest.fixture
def cat_log(spark):
    # assume item 1 is an apple-banana mix and item 2 is a banana
    return spark.createDataFrame(
        data=[
            [1, 1, "red_apples", 5],
            [1, 2, "bananas", 1],
            [2, 1, "healthy_food", 3],
            [3, 1, "bananas", 2],
        ],
        schema="user_idx int, item_idx int, category string, relevance int",
    )


@pytest.fixture
def requested_cats(spark):
    return spark.createDataFrame(
        data=[
            ["healthy_food"],
            ["fruits"],
            ["red_apples"],
        ],
        schema="category string",
    )


@pytest.fixture
def model(cat_tree):
    return CatPopRec(cat_tree)


def test_cat_tree(model):
    mapping = model.leaf_cat_mapping.orderBy("category")
    mapping.show()
    assert mapping.count() == 8
    assert mapping.filter(sf.col("category") == "healthy_food").count() == 1
    assert (
        mapping.filter(sf.col("category") == "healthy_food")
        .select("leaf_cat")
        .collect()[0][0]
        == "healthy_food"
    )

    assert mapping.filter(sf.col("category") == "groceries").count() == 2
    assert sorted(
        mapping.filter(sf.col("category") == "groceries")
        .select("leaf_cat")
        .toPandas()["leaf_cat"]
        .tolist()
    ) == ["bananas", "red_apples"]


def test_works_no_rel(spark, cat_log, requested_cats, model):
    ground_thuth = spark.createDataFrame(
        data=[
            ["red_apples", 1, 1.0],
            ["healthy_food", 1, 1.0],
            ["fruits", 1, 2 / 3],
            ["fruits", 2, 1 / 3],
        ],
        schema="category string, item_idx int, relevance double",
    )
    model.fit(cat_log.drop("relevance"))
    sparkDataFrameEqual(model.predict(requested_cats, k=3), ground_thuth)


def test_works_rel(spark, cat_log, requested_cats, model):
    ground_thuth = spark.createDataFrame(
        data=[
            ["red_apples", 1, 1.0],
            ["healthy_food", 1, 1.0],
            ["fruits", 1, 7 / 8],
            ["fruits", 2, 1 / 8],
        ],
        schema="category string, item_idx int, relevance double",
    )
    model.fit(cat_log)
    sparkDataFrameEqual(model.predict(requested_cats, k=3), ground_thuth)
