# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Dict, List, Tuple

from secretflow.component.checkpoint import CompCheckpoint
from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import (
    DistDataType,
    generate_random_string,
    get_model_public_info,
    load_table,
    model_dumps,
    model_loads,
    save_prediction_dd,
)
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU, SPUObject
from secretflow.device.driver import reveal
from secretflow.ml.linear import SSGLM
from secretflow.ml.linear.ss_glm.core import get_link, Linker
from secretflow.ml.linear.ss_glm.model import STOPPING_METRICS
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab, Table

ss_glm_train_comp = Component(
    "ss_glm_train",
    domain="ml.train",
    version="0.0.2",
    desc="""generalized linear model (GLM) is a flexible generalization of ordinary linear regression.
    The GLM generalizes linear regression by allowing the linear model to be related to the response
    variable via a link function and by allowing the magnitude of the variance of each measurement to
    be a function of its predicted value.""",
)
ss_glm_train_comp.int_attr(
    name="epochs",
    desc="The number of complete pass through the training data.",
    is_list=False,
    is_optional=True,
    default_value=10,
    allowed_values=None,
    lower_bound=1,
    upper_bound=None,
    lower_bound_inclusive=True,
)
ss_glm_train_comp.float_attr(
    name="learning_rate",
    desc="The step size at each iteration in one iteration.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    lower_bound_inclusive=False,
)
ss_glm_train_comp.int_attr(
    name="batch_size",
    desc="The number of training examples utilized in one iteration.",
    is_list=False,
    is_optional=True,
    default_value=1024,
    lower_bound=0,
    lower_bound_inclusive=False,
)
ss_glm_train_comp.str_attr(
    name="link_type",
    desc="link function type",
    is_list=False,
    is_optional=False,
    allowed_values=["Logit", "Log", "Reciprocal", "Identity"],
)
ss_glm_train_comp.str_attr(
    name="label_dist_type",
    desc="label distribution type",
    is_list=False,
    is_optional=False,
    allowed_values=["Bernoulli", "Poisson", "Gamma", "Tweedie"],
)
ss_glm_train_comp.float_attr(
    name="tweedie_power",
    desc="Tweedie distribution power parameter",
    is_list=False,
    is_optional=True,
    default_value=1,
    lower_bound=0,
    lower_bound_inclusive=True,
    upper_bound=2,
    upper_bound_inclusive=True,
)
ss_glm_train_comp.float_attr(
    name="dist_scale",
    desc="A guess value for distribution's scale",
    is_list=False,
    is_optional=True,
    default_value=1,
    lower_bound=1,
    lower_bound_inclusive=True,
)


ss_glm_train_comp.int_attr(
    name="iter_start_irls",
    desc="""run a few rounds of IRLS training as the initialization of w,
    0 disable""",
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=0,
    lower_bound_inclusive=True,
)
ss_glm_train_comp.int_attr(
    name="decay_epoch",
    desc="""decay learning interval""",
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=0,
    lower_bound_inclusive=True,
)
ss_glm_train_comp.float_attr(
    name="decay_rate",
    desc="""decay learning rate""",
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=0,
    lower_bound_inclusive=True,
    upper_bound=1,
    upper_bound_inclusive=False,
)
ss_glm_train_comp.str_attr(
    name="optimizer",
    desc="which optimizer to use: IRLS(Iteratively Reweighted Least Squares) or SGD(Stochastic Gradient Descent)",
    is_list=False,
    is_optional=False,
    allowed_values=["SGD", "IRLS"],
)
ss_glm_train_comp.float_attr(
    name="l2_lambda",
    desc="L2 regularization term",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    lower_bound_inclusive=True,
)

ss_glm_train_comp.int_attr(
    name="infeed_batch_size_limit",
    desc="""size of a single block, default to 10w * 100. increase the size will increase memory cost,
        but may decrease running time. Suggested to be as large as possible. (too large leads to OOM) """,
    is_list=False,
    is_optional=True,
    default_value=10000000,
    lower_bound=1000,
    lower_bound_inclusive=True,
)


ss_glm_train_comp.float_attr(
    name="fraction_of_validation_set",
    desc="fraction of training set to be used as the validation set. ineffective for 'weight' stopping_metric",
    is_list=False,
    is_optional=True,
    default_value=0.2,
    lower_bound=0,
    lower_bound_inclusive=False,
    upper_bound=1,
    upper_bound_inclusive=False,
)

ss_glm_train_comp.int_attr(
    name="random_state",
    desc="""random state for validation split""",
    is_list=False,
    is_optional=True,
    default_value=1212,
    lower_bound=0,
    lower_bound_inclusive=True,
)

ss_glm_train_comp.str_attr(
    name="stopping_metric",
    desc=f"""use what metric as the condition for early stop?  Must be one of {STOPPING_METRICS}.
    only logit link supports AUC metric (note that AUC is very, very expansive in MPC)
    """,
    is_list=False,
    is_optional=True,
    default_value='deviance',
    allowed_values=STOPPING_METRICS,
)

ss_glm_train_comp.int_attr(
    name="stopping_rounds",
    desc="""If the model is not improving for stopping_rounds, the training process will be stopped,
            for 'weight' stopping metric, stopping_rounds is fixed to be 1
    """,
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=0,
    lower_bound_inclusive=True,
    upper_bound=100,
    upper_bound_inclusive=True,
)
ss_glm_train_comp.float_attr(
    name="stopping_tolerance",
    desc="""the model is considered as not improving, if the metric is not improved by tolerance over best metric in history.
    If metric is 'weight' and tolerance == 0, then early stop is disabled.
    """,
    is_list=False,
    is_optional=True,
    default_value=0.001,
    lower_bound=0,
    lower_bound_inclusive=True,
    upper_bound=1,
    upper_bound_inclusive=False,
)

ss_glm_train_comp.bool_attr(
    name="report_metric",
    desc="""Whether to report the value of stopping metric.
    Only effective if early stop is enabled.
    If this option is set to true, metric will be revealed and logged.""",
    is_list=False,
    is_optional=True,
    default_value=False,
)

ss_glm_train_comp.bool_attr(
    name="report_weights",
    desc="If this option is set to true, model will be revealed and model details are visible to all parties",
    is_list=False,
    is_optional=True,
    default_value=False,
)
ss_glm_train_comp.io(
    io_type=IoType.INPUT,
    name="train_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="feature_selects",
            desc="which features should be used for training.",
            col_min_cnt_inclusive=1,
        ),
        TableColParam(
            name="offset",
            desc="Specify a column to use as the offset",
            col_min_cnt_inclusive=0,
            col_max_cnt_inclusive=1,
        ),
        TableColParam(
            name="weight",
            desc="Specify a column to use for the observation weights",
            col_min_cnt_inclusive=0,
            col_max_cnt_inclusive=1,
        ),
        TableColParam(
            name="label",
            desc="Label of train dataset.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        ),
    ],
)
ss_glm_train_comp.io(
    io_type=IoType.OUTPUT,
    name="output_model",
    desc="Output model.",
    types=[DistDataType.SS_GLM_MODEL],
)
ss_glm_train_comp.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="If report_weights is true, report model details",
    types=[DistDataType.REPORT],
)

# current version 0.3
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 3


@ss_glm_train_comp.enable_checkpoint
class SSGLMCheckpoint(CompCheckpoint):
    def associated_arg_names(self) -> List[str]:
        return [
            "epochs",
            "learning_rate",
            "batch_size",
            "link_type",
            "label_dist_type",
            "tweedie_power",
            "dist_scale",
            "iter_start_irls",
            "optimizer",
            "l2_lambda",
            "fraction_of_validation_set",
            "random_state",
            "stopping_metric",
            "stopping_rounds",
            "stopping_tolerance",
            "decay_epoch",
            "decay_rate",
            "train_dataset",
            "train_dataset_label",
            "train_dataset_offset",
            "train_dataset_weight",
            "train_dataset_feature_selects",
        ]


def dump_ss_glm_checkpoint(
    ctx,
    uri: str,
    checkpoint: Tuple[Dict, List[SPUObject]],
    system_info,
) -> DistData:
    train_state, spu_objs = checkpoint
    return model_dumps(
        ctx,
        "ss_glm",
        DistDataType.SS_GLM_CHECKPOINT,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        spu_objs,
        json.dumps(train_state),
        uri,
        system_info,
    )


def load_ss_glm_checkpoint(
    ctx,
    cp: DistData,
    spu: SPU,
) -> Tuple[Dict, List[SPUObject]]:
    spu_objs, model_meta_str = model_loads(
        ctx,
        cp,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_GLM_CHECKPOINT,
        spu=spu,
    )
    train_state = json.loads(model_meta_str)

    return train_state, spu_objs


@ss_glm_train_comp.eval_fn
def ss_glm_train_eval_fn(
    *,
    ctx,
    epochs,
    learning_rate,
    batch_size,
    link_type,
    label_dist_type,
    tweedie_power,
    dist_scale,
    iter_start_irls,
    optimizer,
    l2_lambda,
    infeed_batch_size_limit,
    fraction_of_validation_set,
    random_state,
    stopping_metric,
    stopping_rounds,
    stopping_tolerance,
    report_metric,
    report_weights,
    train_dataset_offset,
    train_dataset_weight,
    decay_epoch,
    decay_rate,
    train_dataset,
    train_dataset_label,
    output_model,
    train_dataset_feature_selects,
    report,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    cluster_def = spu_config["cluster_def"].copy()

    # forced to use 128 ring size & 40 fxp
    cluster_def["runtime_config"]["field"] = "FM128"
    cluster_def["runtime_config"]["fxp_fraction_bits"] = 40

    spu = SPU(cluster_def, spu_config["link_desc"])

    checkpoint = None
    if ctx.comp_checkpoint:
        cp_dd = ctx.comp_checkpoint.load()
        if cp_dd:
            checkpoint = load_ss_glm_checkpoint(ctx, cp_dd, spu)

    glm = SSGLM(spu)

    assert len(train_dataset_label) == 1
    assert (
        train_dataset_label[0] not in train_dataset_feature_selects
    ), f"col {train_dataset_label[0]} used in both label and features"

    y = load_table(
        ctx,
        train_dataset,
        load_labels=True,
        load_features=True,
        col_selects=train_dataset_label,
    )

    x = load_table(
        ctx,
        train_dataset,
        load_labels=True,
        load_features=True,
        col_selects=train_dataset_feature_selects,
    )

    if train_dataset_offset:
        assert (
            train_dataset_offset[0] not in train_dataset_feature_selects
        ), f"col {train_dataset_offset[0]} used in both offset and features"
        offset = load_table(
            ctx,
            train_dataset,
            load_labels=True,
            load_features=True,
            col_selects=train_dataset_offset,
        )
        offset_col = train_dataset_offset[0]
    else:
        offset = None
        offset_col = ""

    if train_dataset_weight:
        assert (
            train_dataset_weight[0] not in train_dataset_feature_selects
        ), f"col {train_dataset_weight[0]} used in both weight and features"
        weight = load_table(
            ctx,
            train_dataset,
            load_labels=True,
            load_features=True,
            col_selects=train_dataset_weight,
        )
    else:
        weight = None

    l2_lambda = l2_lambda if l2_lambda > 0 else None

    def epoch_callback(epoch, cp: Tuple[Dict, List[SPUObject]]):
        cp_uri = f"{output_model}_checkpoint_{epoch}"
        cp_dd = dump_ss_glm_checkpoint(ctx, cp_uri, cp, train_dataset.system_info)
        ctx.comp_checkpoint.save(epoch, cp_dd)

    with ctx.tracer.trace_running():
        if optimizer == "SGD":
            if decay_epoch == 0 or decay_rate == 0:
                decay_rate = None
                decay_epoch = None

            glm.fit_sgd(
                x=x,
                y=y,
                offset=offset,
                weight=weight,
                epochs=epochs,
                link=link_type,
                dist=label_dist_type,
                tweedie_power=tweedie_power,
                scale=dist_scale,
                learning_rate=learning_rate,
                batch_size=batch_size,
                iter_start_irls=iter_start_irls,
                decay_epoch=decay_epoch,
                decay_rate=decay_rate,
                l2_lambda=l2_lambda,
                infeed_batch_size_limit=infeed_batch_size_limit,
                fraction_of_validation_set=fraction_of_validation_set,
                random_state=random_state,
                stopping_metric=stopping_metric,
                stopping_rounds=stopping_rounds,
                stopping_tolerance=stopping_tolerance,
                report_metric=report_metric,
                epoch_callback=epoch_callback if ctx.comp_checkpoint else None,
                recovery_checkpoint=checkpoint,
            )
        elif optimizer == "IRLS":
            glm.fit_irls(
                x=x,
                y=y,
                offset=offset,
                weight=weight,
                epochs=epochs,
                link=link_type,
                dist=label_dist_type,
                tweedie_power=tweedie_power,
                scale=dist_scale,
                l2_lambda=l2_lambda,
                infeed_batch_size_limit=infeed_batch_size_limit,
                fraction_of_validation_set=fraction_of_validation_set,
                random_state=random_state,
                stopping_metric=stopping_metric,
                stopping_rounds=stopping_rounds,
                stopping_tolerance=stopping_tolerance,
                report_metric=report_metric,
                epoch_callback=epoch_callback if ctx.comp_checkpoint else None,
                recovery_checkpoint=checkpoint,
            )
        else:
            raise CompEvalError(f"Unknown optimizer {optimizer}")

    feature_names = x.columns
    party_features_length = {
        device.party: len(columns) for device, columns in x.partition_columns.items()
    }

    model_meta = {
        "link": glm.link.link_type().value,
        "dist": glm.dist.dist_type().value,
        "tweedie_power": tweedie_power,
        "y_scale": glm.y_scale,
        "offset_col": offset_col,
        "label_col": train_dataset_label,
        "feature_names": feature_names,
        "party_features_length": party_features_length,
        "model_hash": generate_random_string(next(iter(x.partition_columns.keys()))),
    }

    model_db = model_dumps(
        ctx,
        "ss_glm",
        DistDataType.SS_GLM_MODEL,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        [glm.spu_w],
        json.dumps(model_meta),
        output_model,
        train_dataset.system_info,
    )
    tabs = []
    if report_weights:
        tabs.append(
            Tab(
                name="weights",
                desc="model weights",
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="descriptions",
                                descriptions=build_weight_desc(glm, x),
                            )
                        ],
                    )
                ],
            )
        )

    effective_train = (report_metric == 'weight' and stopping_tolerance > 0) or (
        report_metric != 'weight' and stopping_rounds > 0
    )

    if report_metric and effective_train:
        tabs.append(
            Tab(
                name="metrics",
                desc="metrics of trainig for each epoch",
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="Table",
                                table=build_metric_table(glm),
                            )
                        ],
                    )
                ],
            )
        )
    report_mate = Report(
        name="weights and metrics",
        desc="model weights report and metrics report",
        tabs=tabs,
    )

    report_dd = DistData(
        name=report,
        type=str(DistDataType.REPORT),
        system_info=train_dataset.system_info,
    )
    report_dd.meta.Pack(report_mate)

    return {"output_model": model_db, "report": report_dd}


def build_weight_desc(glm, x):
    weights = list(map(float, list(reveal(glm.spu_w))))
    named_weight = {}
    for _, features in x.partition_columns.items():
        party_weight = weights[: len(features)]
        named_weight.update({f: w for f, w in zip(features, party_weight)})
        weights = weights[len(features) :]
    assert len(weights) == 1

    w_desc = Descriptions(
        items=[
            Descriptions.Item(
                name="_intercept_", type="float", value=Attribute(f=weights[-1])
            ),
            Descriptions.Item(
                name="_y_scale_", type="float", value=Attribute(f=glm.y_scale)
            ),
        ]
        + [
            Descriptions.Item(name=f, type="float", value=Attribute(f=w))
            for f, w in named_weight.items()
        ],
    )
    return w_desc


def build_metric_table(glm):
    metric_logs = glm.train_metric_history
    assert isinstance(metric_logs, list)
    assert len(metric_logs) >= 1, "must train the model for at least 1 round"
    headers, rows = [], []
    for k in metric_logs[0].keys():
        headers.append(Table.HeaderItem(name=k, desc="", type="str"))

    for i, log in enumerate(metric_logs):
        rows.append(
            Table.Row(name=f"{i}", items=[Attribute(s=str(log[k])) for k in log.keys()])
        )

    metric_table = Table(
        name="metrics log",
        desc="metrics for training and validation set at each epoch (indexed from 1)",
        headers=headers,
        rows=rows,
    )
    return metric_table


ss_glm_predict_comp = Component(
    "ss_glm_predict",
    domain="ml.predict",
    version="0.0.2",
    desc="Predict using the SSGLM model.",
)
ss_glm_predict_comp.party_attr(
    name="receiver",
    desc="Party of receiver.",
    list_min_length_inclusive=1,
    list_max_length_inclusive=1,
)
ss_glm_predict_comp.str_attr(
    name="pred_name",
    desc="Column name for predictions.",
    is_list=False,
    is_optional=True,
    default_value="pred",
    allowed_values=None,
)
ss_glm_predict_comp.bool_attr(
    name="save_ids",
    desc=(
        "Whether to save ids columns into output prediction table. "
        "If true, input feature_dataset must contain id columns, and receiver party must be id owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=True,
)
ss_glm_predict_comp.bool_attr(
    name="save_label",
    desc=(
        "Whether or not to save real label columns into output pred file. "
        "If true, input feature_dataset must contain label columns and receiver party must be label owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=False,
)
ss_glm_predict_comp.io(
    io_type=IoType.INPUT,
    name="model",
    desc="Input model.",
    types=[DistDataType.SS_GLM_MODEL],
)
ss_glm_predict_comp.io(
    io_type=IoType.INPUT,
    name="feature_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="saved_features",
            desc="which features should be saved with prediction result",
            col_min_cnt_inclusive=0,
        )
    ],
)
ss_glm_predict_comp.io(
    io_type=IoType.OUTPUT,
    name="pred",
    desc="Output prediction.",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)


def load_ss_glm_model(ctx, spu, model) -> Tuple[SPUObject, Linker, float]:
    model_objs, model_meta_str = model_loads(
        ctx,
        model,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_GLM_MODEL,
        spu=spu,
    )
    assert len(model_objs) == 1 and isinstance(
        model_objs[0], SPUObject
    ), f"model_objs {model_objs}, model_meta_str {model_meta_str}"

    model_meta = json.loads(model_meta_str)
    assert (
        isinstance(model_meta, dict)
        and "link" in model_meta
        and "y_scale" in model_meta
    ), f"model meta format err {model_meta}"

    return (
        model_objs[0],
        get_link(model_meta["link"]),
        float(model_meta["y_scale"]),
    )


@ss_glm_predict_comp.eval_fn
def ss_glm_predict_eval_fn(
    *,
    ctx,
    feature_dataset,
    feature_dataset_saved_features,
    model,
    receiver,
    pred_name,
    pred,
    save_ids,
    save_label,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    cluster_def = spu_config["cluster_def"].copy()

    # forced to use 128 ring size & 40 fxp
    cluster_def["runtime_config"]["field"] = "FM128"
    cluster_def["runtime_config"]["fxp_fraction_bits"] = 40

    spu = SPU(cluster_def, spu_config["link_desc"])

    model_public_info = get_model_public_info(model)

    glm = SSGLM(spu)
    glm.spu_w, glm.link, glm.y_scale = load_ss_glm_model(ctx, spu, model)

    x = load_table(
        ctx,
        feature_dataset,
        partitions_order=list(model_public_info["party_features_length"].keys()),
        load_features=True,
        col_selects=model_public_info['feature_names'],
    )
    assert x.columns == model_public_info["feature_names"]

    offset_col = model_public_info['offset_col']

    if offset_col:
        offset = load_table(
            ctx,
            feature_dataset,
            load_labels=True,
            load_features=True,
            col_selects=[offset_col],
        )
    else:
        offset = None

    receiver_pyu = PYU(receiver[0])
    with ctx.tracer.trace_running():
        pyu_y = glm.predict(
            x=x,
            o=offset,
            to_pyu=receiver_pyu,
        )

    with ctx.tracer.trace_io():
        y_db = save_prediction_dd(
            ctx,
            pred,
            receiver_pyu,
            pyu_y,
            pred_name,
            feature_dataset,
            feature_dataset_saved_features,
            model_public_info['label_col'] if save_label else [],
            save_ids,
        )

    return {"pred": y_db}
