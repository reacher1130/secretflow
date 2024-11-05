# Copyright 2024 Ant Group Co., Ltd.
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

import math

from google.protobuf.json_format import ParseDict
import pytest

from secretflow.component.component import (
    CompDeclError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import DistDataType
from secretflow.spec.v1.component_pb2 import AttributeDef, ComponentDef, IoDef
from secretflow.spec.v1.data_pb2 import DistData, StorageConfig
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_float_attr():
    comp = Component("test")
    comp.float_attr(
        name="train_size",
        desc="proportion of the dataset to include in the train split.",
        is_list=False,
        is_optional=True,
        default_value=0.75,
        lower_bound=0.0,
        upper_bound=1.0,
        lower_bound_inclusive=True,
        upper_bound_inclusive=True,
    )

    msg = comp._Component__comp_attr_decls[0]
    expected_msg = ParseDict(
        {
            "name": "train_size",
            "desc": "proportion of the dataset to include in the train split.",
            "type": "AT_FLOAT",
            "atomic": {
                "is_optional": True,
                "default_value": {"f": 0.75},
                "lower_bound_enabled": True,
                "lower_bound_inclusive": True,
                "lower_bound": {"f": 0.0},
                "upper_bound_enabled": True,
                "upper_bound_inclusive": True,
                "upper_bound": {"f": 1.0},
            },
        },
        AttributeDef(),
    )
    assert msg == expected_msg


def test_int_attr():
    comp = Component("test")
    comp.int_attr(
        name="epochs",
        is_list=False,
        desc="iteration rounds.",
        is_optional=False,
        default_value=1,
        lower_bound_inclusive=True,
        lower_bound=1,
    )

    msg = comp._Component__comp_attr_decls[0]
    expected_msg = ParseDict(
        {
            "name": "epochs",
            "desc": "iteration rounds.",
            "type": "AT_INT",
            "atomic": {
                "default_value": {"i64": 1},
                "lower_bound_enabled": True,
                "lower_bound_inclusive": True,
                "lower_bound": {"i64": 1},
            },
        },
        AttributeDef(),
    )
    assert msg == expected_msg


def test_str_attr():
    comp = Component("test")
    comp.str_attr(
        name="sig_type",
        is_list=False,
        desc="sigmoid approximation type.",
        is_optional=True,
        default_value="t1",
        allowed_values=["real", "t1", "t3", "t5", "df", "sr", "mix"],
    )

    msg = comp._Component__comp_attr_decls[0]
    expected_msg = ParseDict(
        {
            "name": "sig_type",
            "desc": "sigmoid approximation type.",
            "type": "AT_STRING",
            "atomic": {
                "is_optional": True,
                "default_value": {"s": "t1"},
                "allowed_values": {"ss": ["real", "t1", "t3", "t5", "df", "sr", "mix"]},
            },
        },
        AttributeDef(),
    )
    assert msg == expected_msg


def test_col_params_attr():
    comp = Component("test")
    comp.io(
        io_type=IoType.INPUT,
        name="input_data",
        desc="input for receiver",
        types=[DistDataType.INDIVIDUAL_TABLE],
        col_params=[TableColParam(name="key", desc="Column(s) used to join.")],
    )
    comp.col_params_attr(
        name="observe_feature",
        desc="stratify sample observe feature.",
        table_id="input_data",
        list_min_length_inclusive=1,
        list_max_length_inclusive=1,
    )
    msg = comp._Component__comp_attr_decls[0]
    expected_msg = ParseDict(
        {
            "name": "observe_feature",
            "desc": "stratify sample observe feature.",
            "type": "AT_COL_PARAMS",
            "col_params_binded_table": "input_data",
            "atomic": {
                "list_min_length_inclusive": 1,
                "list_max_length_inclusive": 1,
            },
        },
        AttributeDef(),
    )
    assert msg == expected_msg


def test_col_params_attr_fail():
    comp = Component("test")
    comp.col_params_attr(
        name="observe_feature",
        desc="stratify sample observe feature.",
        table_id="input_data",
        list_min_length_inclusive=1,
        list_max_length_inclusive=1,
    )
    with pytest.raises(
        CompDeclError,
        match=f"table_id input_data is not defined correctly in input.",
    ):
        comp.definition()


def test_col_params_attr_fail_not_table():
    comp = Component("test")
    comp.io(
        io_type=IoType.INPUT,
        name="input_data",
        desc="input for receiver",
        types=[DistDataType.NULL],
        col_params=[TableColParam(name="key", desc="Column(s) used to join.")],
    )
    comp.col_params_attr(
        name="observe_feature",
        desc="stratify sample observe feature.",
        table_id="input_data",
        list_min_length_inclusive=1,
        list_max_length_inclusive=1,
    )
    with pytest.raises(
        CompDeclError,
        match=f"table_id input_data is not defined correctly in input.",
    ):
        comp.definition()


def test_col_params_attr_fail_not_all_table():
    comp = Component("test")
    comp.io(
        io_type=IoType.INPUT,
        name="input_data",
        desc="input for receiver",
        types=[DistDataType.NULL, DistDataType.INDIVIDUAL_TABLE],
        col_params=[TableColParam(name="key", desc="Column(s) used to join.")],
    )
    comp.col_params_attr(
        name="observe_feature",
        desc="stratify sample observe feature.",
        table_id="input_data",
        list_min_length_inclusive=1,
        list_max_length_inclusive=1,
    )
    with pytest.raises(
        CompDeclError,
        match=f"table_id input_data is not defined correctly in input.",
    ):
        comp.definition()


def test_bool_attr():
    comp = Component("test")
    comp.bool_attr(
        name="shuffle",
        is_list=False,
        desc="Whether or not to shuffle the data before splitting.",
        is_optional=True,
        default_value=True,
    )

    def f():
        pass

    msg = comp._Component__comp_attr_decls[0]
    expected_msg = ParseDict(
        {
            "name": "shuffle",
            "desc": "Whether or not to shuffle the data before splitting.",
            "type": "AT_BOOL",
            "atomic": {
                "is_optional": True,
                "default_value": {"b": True},
            },
        },
        AttributeDef(),
    )
    assert msg == expected_msg


def test_struct_attr_group():
    comp = Component("test")
    comp.struct_attr_group(
        name="level0",
        desc="level0",
        group=[
            comp.bool_attr(
                name="bool0",
                desc="",
                is_list=False,
                is_optional=True,
                default_value=True,
            ),
            comp.struct_attr_group(
                name="level1",
                desc="level1",
                group=[
                    comp.bool_attr(
                        name="bool1",
                        desc="",
                        is_list=False,
                        is_optional=True,
                        default_value=True,
                    ),
                    comp.bool_attr(
                        name="bool2",
                        desc="",
                        is_list=False,
                        is_optional=True,
                        default_value=True,
                    ),
                ],
            ),
        ],
    )

    msg = comp.definition()

    expected_msg = ParseDict(
        {
            "name": "test",
            "attrs": [
                {"name": "level0", "desc": "level0", "type": "AT_STRUCT_GROUP"},
                {
                    "prefixes": ["level0"],
                    "name": "bool0",
                    "type": "AT_BOOL",
                    "atomic": {"isOptional": True, "defaultValue": {"b": True}},
                },
                {
                    "prefixes": ["level0"],
                    "name": "level1",
                    "desc": "level1",
                    "type": "AT_STRUCT_GROUP",
                },
                {
                    "prefixes": ["level0", "level1"],
                    "name": "bool1",
                    "type": "AT_BOOL",
                    "atomic": {"isOptional": True, "defaultValue": {"b": True}},
                },
                {
                    "prefixes": ["level0", "level1"],
                    "name": "bool2",
                    "type": "AT_BOOL",
                    "atomic": {"isOptional": True, "defaultValue": {"b": True}},
                },
            ],
        },
        ComponentDef(),
    )
    assert msg == expected_msg


def test_union_attr_group():
    comp = Component("test")
    comp.union_attr_group(
        name="level0",
        desc="",
        group=[
            comp.union_selection_attr(
                name="dummy1a",
                desc="",
            ),
            comp.struct_attr_group(
                name="level1b",
                desc="",
                group=[
                    comp.bool_attr(
                        name="bool2",
                        desc="",
                        is_list=False,
                        is_optional=True,
                        default_value=True,
                    ),
                ],
            ),
            comp.union_attr_group(
                name="level1c",
                desc="",
                group=[
                    comp.struct_attr_group(
                        name="level2a",
                        desc="",
                        group=[
                            comp.bool_attr(
                                name="bool3",
                                desc="",
                                is_list=False,
                                is_optional=True,
                                default_value=True,
                            )
                        ],
                    ),
                    comp.union_selection_attr(
                        name="dummy2b",
                        desc="",
                    ),
                    comp.struct_attr_group(
                        name="level2c",
                        desc="",
                        group=[
                            comp.bool_attr(
                                name="bool3",
                                desc="",
                                is_list=False,
                                is_optional=True,
                                default_value=True,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    msg = comp.definition()

    expected_msg = ParseDict(
        {
            "name": "test",
            "attrs": [
                {
                    "name": "level0",
                    "type": "AT_UNION_GROUP",
                    "union": {"defaultSelection": "dummy1a"},
                },
                {
                    "prefixes": ["level0"],
                    "name": "dummy1a",
                    "type": "ATTR_TYPE_UNSPECIFIED",
                },
                {
                    "prefixes": ["level0"],
                    "name": "level1b",
                    "type": "AT_STRUCT_GROUP",
                },
                {
                    "prefixes": ["level0", "level1b"],
                    "name": "bool2",
                    "type": "AT_BOOL",
                    "atomic": {"isOptional": True, "defaultValue": {"b": True}},
                },
                {
                    "prefixes": ["level0"],
                    "name": "level1c",
                    "type": "AT_UNION_GROUP",
                    "union": {"defaultSelection": "level2a"},
                },
                {
                    "prefixes": ["level0", "level1c"],
                    "name": "level2a",
                    "type": "AT_STRUCT_GROUP",
                },
                {
                    "prefixes": ["level0", "level1c", "level2a"],
                    "name": "bool3",
                    "type": "AT_BOOL",
                    "atomic": {"isOptional": True, "defaultValue": {"b": True}},
                },
                {
                    "prefixes": ["level0", "level1c"],
                    "name": "dummy2b",
                    "type": "ATTR_TYPE_UNSPECIFIED",
                },
                {
                    "prefixes": ["level0", "level1c"],
                    "name": "level2c",
                    "type": "AT_STRUCT_GROUP",
                },
                {
                    "prefixes": ["level0", "level1c", "level2c"],
                    "name": "bool3",
                    "type": "AT_BOOL",
                    "atomic": {"isOptional": True, "defaultValue": {"b": True}},
                },
            ],
        },
        ComponentDef(),
    )
    assert msg == expected_msg


def test_table_io():
    comp = Component("test")
    comp.io(
        io_type=IoType.INPUT,
        name="receiver_input",
        desc="input for receiver",
        types=[DistDataType.INDIVIDUAL_TABLE],
        col_params=[TableColParam(name="key", desc="Column(s) used to join.")],
    )

    msg = comp._Component__input_io_decls[0]
    expected_msg = ParseDict(
        {
            "name": "receiver_input",
            "desc": "input for receiver",
            "types": [str(DistDataType.INDIVIDUAL_TABLE)],
            "attrs": [
                {
                    "name": "key",
                    "desc": "Column(s) used to join.",
                }
            ],
        },
        IoDef(),
    )
    assert msg == expected_msg


def test_model_io():
    comp = Component("test")
    comp.io(
        io_type=IoType.OUTPUT,
        name="output_model",
        desc="desc",
        types=[DistDataType.SS_SGD_MODEL],
    )

    msg = comp._Component__output_io_decls[0]
    expected_msg = ParseDict(
        {
            "name": "output_model",
            "desc": "desc",
            "types": [str(DistDataType.SS_SGD_MODEL)],
        },
        IoDef(),
    )
    assert msg == expected_msg


def test_definition():
    comp = Component("comp_a", domain="sf", version="0.0.2b1", desc="desc of comp_a")
    comp.float_attr(
        name="train_size",
        desc="proportion of the dataset to include in the train split.",
        is_list=False,
        is_optional=True,
        default_value=0.75,
        lower_bound=0.0,
        upper_bound=1.0,
        lower_bound_inclusive=True,
        upper_bound_inclusive=True,
    )
    comp.io(
        io_type=IoType.INPUT,
        name="receiver_input",
        desc="input for receiver",
        types=[DistDataType.INDIVIDUAL_TABLE],
        col_params=[
            TableColParam(
                name="key",
                desc="Column(s) used to join.",
                col_min_cnt_inclusive=1,
                col_max_cnt_inclusive=3,
            )
        ],
    )
    comp.io(
        io_type=IoType.OUTPUT,
        name="output_model",
        desc="desc",
        types=[DistDataType.SS_SGD_MODEL],
    )

    expected_msg = ParseDict(
        {
            "domain": "sf",
            "name": "comp_a",
            "desc": "desc of comp_a",
            "version": "0.0.2b1",
            "attrs": [
                {
                    "name": "train_size",
                    "desc": "proportion of the dataset to include in the train split.",
                    "type": "AT_FLOAT",
                    "atomic": {
                        "is_optional": True,
                        "default_value": {"f": 0.75},
                        "lower_bound_enabled": True,
                        "lower_bound_inclusive": True,
                        "lower_bound": {"f": 0.0},
                        "upper_bound_enabled": True,
                        "upper_bound_inclusive": True,
                        "upper_bound": {"f": 1.0},
                    },
                }
            ],
            "inputs": [
                {
                    "name": "receiver_input",
                    "desc": "input for receiver",
                    "types": [str(DistDataType.INDIVIDUAL_TABLE)],
                    "attrs": [
                        {
                            "name": "key",
                            "desc": "Column(s) used to join.",
                            "col_min_cnt_inclusive": 1,
                            "col_max_cnt_inclusive": 3,
                        }
                    ],
                }
            ],
            "outputs": [
                {
                    "name": "output_model",
                    "desc": "desc",
                    "types": [str(DistDataType.SS_SGD_MODEL)],
                }
            ],
        },
        ComponentDef(),
    )
    assert comp.definition() == expected_msg


def test_eval():
    comp = Component("test")
    comp.float_attr(
        name="train_size",
        desc="proportion of the dataset to include in the train split.",
        is_list=False,
        is_optional=True,
        default_value=0.75,
        lower_bound=0.0,
        upper_bound=1.0,
        lower_bound_inclusive=True,
        upper_bound_inclusive=True,
    )
    comp.io(
        io_type=IoType.OUTPUT,
        name="output_model",
        desc="desc",
        types=[DistDataType.SS_SGD_MODEL],
    )
    comp.io(
        io_type=IoType.INPUT,
        name="receiver_input",
        desc="input for receiver",
        types=[DistDataType.INDIVIDUAL_TABLE],
        col_params=[
            TableColParam(
                name="key",
                desc="Column(s) used to join.",
                col_min_cnt_inclusive=1,
                col_max_cnt_inclusive=3,
            )
        ],
    )

    @comp.eval_fn
    def fn(*, ctx, train_size, output_model, receiver_input, receiver_input_key):
        assert math.isclose(train_size, 0.8, rel_tol=0.001)
        assert output_model == "test"
        assert receiver_input.name == "receiver_input"
        assert receiver_input_key == ["a", "b", "c"]
        return {
            "output_model": DistData(
                name=str(train_size), type=str(DistDataType.SS_SGD_MODEL)
            )
        }

    node_json = {
        "name": "test",
        "attr_paths": ["train_size", "input/receiver_input/key"],
        "attrs": [{"f": 0.8}, {"ss": ["a", "b", "c"]}],
        "output_uris": ["test"],
        "inputs": [
            {"name": "receiver_input", "type": str(DistDataType.INDIVIDUAL_TABLE)}
        ],
    }

    node = ParseDict(node_json, NodeEvalParam())
    storage_config = StorageConfig(
        type="local_fs",
        local_fs=StorageConfig.LocalFSConfig(wd=f"/tmp/tmp"),
    )
    ret = float(comp.eval(node, storage_config).outputs[0].name)

    assert math.isclose(ret, 0.8, rel_tol=0.001)
