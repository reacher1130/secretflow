



SecretFlow Component List
=========================


Last update: Fri Jul 12 11:12:51 2024

Version: 0.0.1

First-party SecretFlow components.
## data_filter

### condition_filter


Component version: 0.0.1

Filter the table based on a single column's values and condition.
Warning: the party responsible for condition filtering will directly send the sample distribution to other participants.
Malicious participants can obtain the distribution of characteristics by repeatedly calling with different filtering values.
Audit the usage of this component carefully.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|comparator|Comparator to use for comparison. Must be one of '==','<','<=','>','>=','IN'|String|Y|Allowed: ['==', '<', '<=', '>', '>=', 'IN'].|
|value_type|Type of the value to compare with. Must be one of ['STRING', 'FLOAT']|String|Y|Allowed: ['STRING', 'FLOAT'].|
|bound_value|Input a str with values separated by ','. List of values to compare with. If comparator is not 'IN', we only support one element in this list.|String|Y||
|float_epsilon|Epsilon value for floating point comparison. WARNING: due to floating point representation in computers, set this number slightly larger if you want filter out the values exactly at desired boundary. for example, abs(1.001 - 1.002) is slightly larger than 0.001, and therefore may not be filter out using == and epsilson = 0.001|Float|N|Default: 0.0.Range: [0.0, $\infty$).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|in_ds|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/in_ds/features|Feature(s) to operate on.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|out_ds|Output vertical table that satisfies the condition.|['sf.table.vertical_table']||
|out_ds_else|Output vertical table that does not satisfies the condition.|['sf.table.vertical_table']||

### expr_condition_filter


Component version: 0.0.1

Only row-level filtering is supported, column processing is not available;
the custom expression must comply with SQLite syntax standards
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|expr|The custom expression must comply with SQLite syntax standards|String|Y||

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|in_ds|Input vertical or individual table|['sf.table.individual', 'sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|out_ds|Output table that satisfies the condition|['sf.table.individual', 'sf.table.vertical_table']||
|out_ds_else|Output table that does not satisfies the condition|['sf.table.individual', 'sf.table.vertical_table']||

### feature_filter


Component version: 0.0.1

Drop features from the dataset.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|in_ds|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/in_ds/drop_features|Features to drop.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|out_ds|Output vertical table.|['sf.table.vertical_table']||

### sample


Component version: 0.0.1

Sample data set.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|sample_algorithm|sample algorithm and parameters|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|sample_algorithm/random|Random sample.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|sample_algorithm/random/frac|Proportion of the dataset to sample in the set. The fraction should be larger than 0.|Float|N|Default: 0.8.Range: (0.0, 10000.0).|
|sample_algorithm/random/random_state|Specify the random seed of the shuffling.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|sample_algorithm/random/replacement|If true, sampling with replacement. If false, sampling without replacement.|Boolean|N|Default: False.|
|sample_algorithm/system|system sample.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|sample_algorithm/system/frac|Proportion of the dataset to sample in the set. The fraction should be larger than 0 and less than or equal to 0.5.|Float|N|Default: 0.2.Range: (0.0, 0.5].|
|sample_algorithm/stratify|stratify sample.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|sample_algorithm/stratify/frac|Proportion of the dataset to sample in the set. The fraction should be larger than 0.|Float|N|Default: 0.8.Range: (0.0, 10000.0).|
|sample_algorithm/stratify/random_state|Specify the random seed of the shuffling.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|sample_algorithm/stratify/observe_feature|stratify sample observe feature.|String|Y||
|sample_algorithm/stratify/replacements|If true, sampling with replacement. If false, sampling without replacement.|Boolean List|Y||
|sample_algorithm/stratify/quantiles|stratify sample quantiles|Float List|Y|Min length(inclusive): 1. Max length(inclusive): 1000.|
|sample_algorithm/stratify/weights|stratify sample weights|Float List|N|Default: [].Range: ([], []).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table', 'sf.table.individual']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|sample_output|Output sampled dataset.|['sf.table.vertical_table', 'sf.table.individual']||
|reports|Output sample reports|['sf.report']||

## data_prep

### psi


Component version: 0.0.5

PSI between two parties.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|protocol|PSI protocol.|String|N|Default: PROTOCOL_RR22.Allowed: ['PROTOCOL_RR22', 'PROTOCOL_ECDH', 'PROTOCOL_KKRT'].|
|sort_result|It false, output is not promised to be aligned. Warning: disable this option may lead to errors in the following components. DO NOT TURN OFF if you want to append other components.|Boolean|N|Default: True.|
|allow_duplicate_keys|Some join types allow duplicate keys.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|allow_duplicate_keys/no|Duplicate keys are not allowed.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|allow_duplicate_keys/no/skip_duplicates_check|If true, the check of duplicated items will be skiped.|Boolean|N|Default: False.|
|allow_duplicate_keys/no/check_hash_digest|Check if hash digest of keys from parties are equal to determine whether to early-stop.|Boolean|N|Default: False.|
|allow_duplicate_keys/yes|Duplicate keys are allowed.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|allow_duplicate_keys/yes/join_type|Join type.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|allow_duplicate_keys/yes/join_type/left_join|Left join with duplicate keys|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|allow_duplicate_keys/yes/join_type/left_join/left_side|Required for left join|Special type. Specify parties.|Y||
|fill_value_int|For int type data. Use this value for filling null.|Integer|N|Default: 0.|
|ecdh_curve|Curve type for ECDH PSI.|String|N|Default: CURVE_FOURQ.Allowed: ['CURVE_25519', 'CURVE_FOURQ', 'CURVE_SM2', 'CURVE_SECP256K1'].|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|receiver_input|Individual table for receiver|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/receiver_input/key|Column(s) used to join.|String List(Set value with other Component Attributes)|You need to select some columns of table receiver_input. Min column number to select(inclusive): 1. |
|sender_input|Individual table for sender|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/sender_input/key|Column(s) used to join.|String List(Set value with other Component Attributes)|You need to select some columns of table sender_input. Min column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|psi_output|Output vertical table|['sf.table.vertical_table']||

### train_test_split


Component version: 0.0.1

Split datasets into random train and test subsets.
- Please check: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|train_size|Proportion of the dataset to include in the train subset. The sum of test_size and train_size should be in the (0, 1] range.|Float|N|Default: 0.75.Range: (0.0, 1.0).|
|test_size|Proportion of the dataset to include in the test subset. The sum of test_size and train_size should be in the (0, 1] range.|Float|N|Default: 0.25.Range: (0.0, 1.0).|
|random_state|Specify the random seed of the shuffling.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|shuffle|Whether to shuffle the data before splitting.|Boolean|N|Default: True.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train|Output train dataset.|['sf.table.vertical_table']||
|test|Output test dataset.|['sf.table.vertical_table']||

### union


Component version: 0.0.1

Perform a horizontal merge of two data tables, supporting the individual table or vertical table on the same node.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input1|The first input table|['sf.table.individual', 'sf.table.vertical_table']||
|input2|The second input table|['sf.table.individual', 'sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output table|['sf.table.individual', 'sf.table.vertical_table']||

## feature

### vert_binning


Component version: 0.0.2

Generate equal frequency or equal range binning rules for vertical partitioning datasets.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|binning_method|How to bin features with numeric types: "quantile"(equal frequency)/"eq_range"(equal range)|String|N|Default: eq_range.Allowed: ['eq_range', 'quantile'].|
|bin_num|Max bin counts for one features.|Integer|N|Default: 10.Range: [2, $\infty$).|
|report_rules|Whether report binning rules.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/input_data/feature_selects|which features should be binned.|String List(Set value with other Component Attributes)|You need to select some columns of table input_data. Min column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|bin_rule|Output bin rule.|['sf.rule.binning']||
|report|report rules details if report_rules is true|['sf.report']||

### vert_woe_binning


Component version: 0.0.2

Generate Weight of Evidence (WOE) binning rules for vertical partitioning datasets.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|secure_device_type|Use SPU(Secure multi-party computation or MPC) or HEU(Homomorphic encryption or HE) to secure bucket summation.|String|N|Default: spu.Allowed: ['spu', 'heu'].|
|binning_method|How to bin features with numeric types: "quantile"(equal frequency)/"chimerge"(ChiMerge from AAAI92-019: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)/"eq_range"(equal range)|String|N|Default: quantile.Allowed: ['quantile', 'chimerge', 'eq_range'].|
|bin_num|Max bin counts for one features.|Integer|N|Default: 10.Range: (0, $\infty$).|
|positive_label|Which value represent positive value in label.|String|N|Default: 1.|
|chimerge_init_bins|Max bin counts for initialization binning in ChiMerge.|Integer|N|Default: 100.Range: (2, $\infty$).|
|chimerge_target_bins|Stop merging if remaining bin counts is less than or equal to this value.|Integer|N|Default: 10.Range: [2, $\infty$).|
|chimerge_target_pvalue|Stop merging if biggest pvalue of remaining bins is greater than this value.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|report_rules|Whether report binning rules.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/input_data/feature_selects|which features should be binned. WARNING: WOE won't be effective for features with enumeration count <=2.|String List(Set value with other Component Attributes)|You need to select some columns of table input_data. Min column number to select(inclusive): 1. |
|input/input_data/label|Label of input data.|String List(Set value with other Component Attributes)|You need to select some columns of table input_data. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|bin_rule|Output WOE rule.|['sf.rule.binning']||
|report|report rules details if report_rules is true|['sf.report']||

## io

### identity


Component version: 0.0.1

map any input to output
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input data|['sf.model.ss_glm', 'sf.model.sgb', 'sf.model.ss_xgb', 'sf.model.ss_sgd', 'sf.rule.binning', 'sf.rule.preprocessing', 'sf.read_data']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_data|Output data|['sf.model.ss_glm', 'sf.model.sgb', 'sf.model.ss_xgb', 'sf.model.ss_sgd', 'sf.rule.binning', 'sf.rule.preprocessing', 'sf.read_data']||

### read_data


Component version: 0.0.1

read model or rules from sf cluster
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_dd|Input dist data|['sf.rule.binning', 'sf.model.ss_glm']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_data|Output rules or models in DistData.meta|['sf.read_data']||

### write_data


Component version: 0.0.1

write model or rules back to sf cluster
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|write_data|rule or model protobuf by json format|String|Y||
|write_data_type|which rule or model is writing|String|N|Default: sf.rule.binning.Allowed: ['sf.rule.binning', 'sf.model.ss_glm'].|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_dd|Input dist data. Rule reconstructions may need hidden info in original rule for security considerations.|['sf.rule.binning', 'sf.model.ss_glm']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output rules or models in sf cluster format|['sf.rule.binning', 'sf.model.ss_glm']||

## ml.eval

### biclassification_eval


Component version: 0.0.1

Statistics evaluation for a bi-classification model on a dataset.
1. summary_report: SummaryReport
2. eq_frequent_bin_report: List[EqBinReport]
3. eq_range_bin_report: List[EqBinReport]
4. head_report: List[PrReport]
reports for fpr = 0.001, 0.005, 0.01, 0.05, 0.1, 0.2
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|bucket_size|Number of buckets.|Integer|N|Default: 10.Range: [1, $\infty$).|
|min_item_cnt_per_bucket|Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 5.|Integer|N|Default: 5.Range: [5, $\infty$).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|in_ds|Input table with prediction and label, usually is a result from a prediction component.|['sf.table.vertical_table', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/in_ds/label|The label name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |
|input/in_ds/prediction|The prediction result column name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|reports|Output report.|['sf.report']||

### prediction_bias_eval


Component version: 0.0.1

Calculate prediction bias, ie. average of predictions - average of labels.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|bucket_num|Num of bucket.|Integer|N|Default: 10.Range: [1, $\infty$).|
|min_item_cnt_per_bucket|Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 2.|Integer|N|Default: 2.Range: [2, $\infty$).|
|bucket_method|Bucket method.|String|N|Default: equal_width.Allowed: ['equal_width', 'equal_frequency'].|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|in_ds|Input table with prediction and label, usually is a result from a prediction component.|['sf.table.vertical_table', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/in_ds/label|The label name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |
|input/in_ds/prediction|The prediction result column name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|result|Output report.|['sf.report']||

### regression_eval


Component version: 0.0.1

Statistics evaluation for a regression model on a dataset.
Contained Statistics:
R2 Score (r2_score): It is a statistical measure that represents the proportion of the variance in the dependent variable that can be predicted from the independent variables. It ranges from 0 to 1, where a higher value indicates a better fit.
Mean Absolute Error (mean_abs_err): It calculates the average absolute difference between the predicted and actual values. It provides a measure of the average magnitude of the errors.
Mean Absolute Percentage Error (mean_abs_percent_err): It calculates the average absolute percentage difference between the predicted and actual values. It measures the average magnitude of the errors in terms of percentages.
Sum of Squared Errors (sum_squared_errors): It calculates the sum of the squared differences between the predicted and actual values. It provides an overall measure of the model's performance.
Mean Squared Error (mean_squared_errors): It calculates the average of the squared differences between the predicted and actual values. It is widely used as a loss function in regression problems.
Root Mean Squared Error (root_mean_squared_errors): It is the square root of the mean squared error. It provides a measure of the average magnitude of the errors in the original scale of the target variable.
Mean of True Values (y_true_mean): It calculates the average of the actual values in the target variable. It can be useful for establishing a baseline for the model's performance.
Mean of Predicted Values (y_pred_mean): It calculates the average of the predicted values. It can be compared with the y_true_mean to get an idea of the model's bias.
Residual Histograms (residual_hists): It represents the distribution of the differences between the predicted and actual values. It helps to understand the spread and pattern of the errors.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|bucket_size|Number of buckets for residual histogram.|Integer|N|Default: 10.Range: [1, 10000].|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|in_ds|Input table with prediction and label, usually is a result from a prediction component.|['sf.table.vertical_table', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/in_ds/label|The label name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |
|input/in_ds/prediction|The prediction result column name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|reports|Output report.|['sf.report']||

### ss_pvalue


Component version: 0.0.1

Calculate P-Value for LR model training on vertical partitioning dataset by using secret sharing.
For large dataset(large than 10w samples & 200 features),
recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.ss_sgd', 'sf.model.ss_glm']||
|input_data|Input vertical table.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output P-Value report.|['sf.report']||

## ml.predict

### sgb_predict


Component version: 0.0.3

Predict using SGB model.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Name for prediction column|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: True.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|model|['sf.model.sgb']||
|feature_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/feature_dataset/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table feature_dataset. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

### slnn_predict


Component version: 0.0.2

Predict using the SLNN model.
This component is not enabled by default, it requires the use of the full version
of secretflow image and setting the ENABLE_NN environment variable to true.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|batch_size|The number of examples per batch.|Integer|N|Default: 8192.Range: (0, $\infty$).|
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.sl_nn']||
|feature_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/feature_dataset/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table feature_dataset. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

### ss_glm_predict


Component version: 0.0.2

Predict using the SSGLM model.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: True.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.ss_glm']||
|feature_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/feature_dataset/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table feature_dataset. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

### ss_sgd_predict


Component version: 0.0.2

Predict using the SS-SGD model.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: True.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.ss_sgd']||
|feature_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/feature_dataset/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table feature_dataset. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

### ss_xgb_predict


Component version: 0.0.2

Predict using the SS-XGB model.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: True.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.ss_xgb']||
|feature_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/feature_dataset/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table feature_dataset. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

## ml.train

### sgb_train


Component version: 0.0.4

Provides both classification and regression tree boosting (also known as GBDT, GBM)
for vertical split dataset setting by using secure boost.
- SGB is short for SecureBoost. Compared to its safer counterpart SS-XGB, SecureBoost focused on protecting label holder.
- Check https://arxiv.org/abs/1901.08755.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|num_boost_round|Number of boosting iterations.|Integer|N|Default: 10.Range: [1, $\infty$).|
|max_depth|Maximum depth of a tree.|Integer|N|Default: 5.Range: [1, 16].|
|learning_rate|Step size shrinkage used in update to prevent overfitting.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|objective|Specify the learning objective.|String|N|Default: logistic.Allowed: ['linear', 'logistic', 'tweedie'].|
|reg_lambda|L2 regularization term on weights.|Float|N|Default: 0.1.Range: [0.0, 10000.0].|
|gamma|Greater than 0 means pre-pruning enabled. If gain of a node is less than this value, it would be pruned.|Float|N|Default: 1.0.Range: [0.0, 10000.0].|
|colsample_by_tree|Subsample ratio of columns when constructing each tree.|Float|N|Default: 1.0.Range: (0.0, 1.0].|
|sketch_eps|This roughly translates into O(1 / sketch_eps) number of bins.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|base_score|The initial prediction score of all instances, global bias.|Float|N|Default: 0.0.Range: [-10.0, 10.0].|
|seed|Pseudorandom number generator seed.|Integer|N|Default: 42.Range: [0, $\infty$).|
|fixed_point_parameter|Any floating point number encoded by heu, will multiply a scale and take the round, scale = 2 ** fixed_point_parameter. larger value may mean more numerical accuracy, but too large will lead to overflow problem.|Integer|N|Default: 20.Range: [1, 100].|
|first_tree_with_label_holder_feature|Whether to train the first tree with label holder's own features.|Boolean|N|Default: False.|
|batch_encoding_enabled|If use batch encoding optimization.|Boolean|N|Default: True.|
|enable_quantization|Whether enable quantization of g and h.|Boolean|N|Default: False.|
|quantization_scale|Scale the sum of g to the specified value.|Float|N|Default: 10000.0.Range: [0.0, 10000000.0].|
|max_leaf|Maximum leaf of a tree. Only effective if train leaf wise.|Integer|N|Default: 15.Range: [1, 32768].|
|rowsample_by_tree|Row sub sample ratio of the training instances.|Float|N|Default: 1.0.Range: (0.0, 1.0].|
|enable_goss|Whether to enable GOSS.|Boolean|N|Default: False.|
|top_rate|GOSS-specific parameter. The fraction of large gradients to sample.|Float|N|Default: 0.3.Range: (0.0, 1.0].|
|bottom_rate|GOSS-specific parameter. The fraction of small gradients to sample.|Float|N|Default: 0.5.Range: (0.0, 1.0].|
|tree_growing_method|How to grow tree?|String|N|Default: level.|
|enable_early_stop|Whether to enable early stop during training.|Boolean|N|Default: False.|
|enable_monitor|Whether to enable monitoring performance during training.|Boolean|N|Default: False.|
|eval_metric|Use what metric for monitoring and early stop? Currently support ['roc_auc', 'rmse', 'mse', 'tweedie_deviance', 'tweedie_nll']|String|N|Default: roc_auc.Allowed: ['roc_auc', 'rmse', 'mse', 'tweedie_deviance', 'tweedie_nll'].|
|validation_fraction|Early stop specific parameter. Only effective if early stop enabled. The fraction of samples to use as validation set.|Float|N|Default: 0.1.Range: (0.0, 1.0).|
|stopping_rounds|Early stop specific parameter. If more than 'stopping_rounds' consecutive rounds without improvement, training will stop. Only effective if early stop enabled|Integer|N|Default: 1.Range: [1, 1024].|
|stopping_tolerance|Early stop specific parameter. If metric on validation set is no longer improving by at least this amount, then consider not improving.|Float|N|Default: 0.0.Range: [0.0, $\infty$).|
|tweedie_variance_power|Parameter that controls the variance of the Tweedie distribution.|Float|N|Default: 1.5.Range: (1.0, 2.0).|
|save_best_model|Whether to save the best model on validation set during training.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/train_dataset/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. |
|input/train_dataset/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.sgb']||

### slnn_train


Component version: 0.0.1

Train nn models for vertical partitioning dataset by split learning.
This component is not enabled by default, it requires the use of the full version
of secretflow image and setting the ENABLE_NN environment variable to true.
Since it is necessary to define the model structure using python code,
although the range of syntax and APIs that can be used has been restricted,
there are still potential security risks. It is recommended to use it in
conjunction with process sandboxes such as nsjail.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|models|Define the models for training.|String|Y||
|epochs|The number of complete pass through the training data.|Integer|N|Default: 10.Range: [1, $\infty$).|
|learning_rate|The step size at each iteration in one iteration.|Float|N|Default: 0.001.Range: (0.0, $\infty$).|
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 512.Range: (0, $\infty$).|
|validattion_prop|The proportion of validation set to total data set.|Float|N|Default: 0.1.Range: [0.0, 1.0).|
|loss|Loss function.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|loss/builtin|Builtin loss function.|String|N|Default: mean_squared_error.Allowed: ['binary_crossentropy', 'categorical_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_similarity', 'huber', 'kl_divergence', 'log_cosh', 'poisson', 'binary_focal_crossentropy', 'sparse_categorical_crossentropy', 'hinge', 'categorical_hinge', 'squared_hinge'].|
|loss/custom|Custom loss function.|String|N|Default: `def loss(y_true, y_pred):\n    return tf.keras.losses.mean_squared_error(y_true, y_pred)\n\n\ncompile_loss(loss)\n\n`.|
|optimizer|Optimizer.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|optimizer/name|Optimizer name.|String|Y|Allowed: ['Adam', 'SGD', 'RMSprop', 'AdamW', 'Adamax', 'Nadam', 'Adagrad', 'Adadelta', 'Adafactor', 'Ftrl', 'Lion'].|
|optimizer/params|Additional optimizer parameters in JSON format.|String|N|Default: .|
|metrics|Metrics.|String List|N|Max length(inclusive): 10. Default: ['AUC'].Allowed: ['AUC', 'Accuracy', 'Precision', 'Recall', 'BinaryAccuracy', 'BinaryCrossentropy', 'CategoricalAccuracy', 'CategoricalCrossentropy', 'CosineSimilarity', 'FalseNegatives', 'FalsePositives', 'TrueNegatives', 'TruePositives', 'KLDivergence', 'LogCoshError', 'MeanAbsoluteError', 'MeanAbsolutePercentageError', 'MeanRelativeError', 'MeanSquaredError', 'MeanSquaredLogarithmicError', 'Hinge', 'SquaredHinge', 'CategoricalHinge', 'BinaryIoU', 'IoU', 'MeanIoU', 'OneHotIoU', 'OneHotMeanIoU', 'Poisson', 'PrecisionAtRecall', 'RecallAtPrecision', 'RootMeanSquaredError', 'SensitivityAtSpecificity', 'SparseCategoricalAccuracy', 'SparseCategoricalCrossentropy', 'SparseTopKCategoricalAccuracy', 'SpecificityAtSensitivity', 'TopKCategoricalAccuracy'].|
|model_input_scheme|Input scheme of base model, tensor: merge all features into one tensor; tensor_dict: each feature as a tensor.|String|Y|Allowed: ['tensor', 'tensor_dict'].|
|strategy|Split learning strategy.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|strategy/name|Split learning strategy name.|String|N|Default: pipeline.Allowed: ['pipeline', 'split_nn', 'split_async', 'split_state_async'].|
|strategy/params|Additional strategy parameters in JSON format.|String|N|Default: {"pipeline_size":2}.|
|compressor|Compressor for hiddens and gradients.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|compressor/name|Compressor name.|String|N|Default: .Allowed: ['', 'topk_sparse', 'random_sparse', 'stc_sparse', 'scr_sparse', 'quantized_fp', 'quantized_lstm', 'quantized_kmeans', 'quantized_zeropoint', 'mixed_compressor'].|
|compressor/params|Additional compressor parameters in JSON format.|String|N|Default: .|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/train_dataset/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. |
|input/train_dataset/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.sl_nn']||
|reports|Output report.|['sf.report']||

### ss_glm_train


Component version: 0.0.3

generalized linear model (GLM) is a flexible generalization of ordinary linear regression.
The GLM generalizes linear regression by allowing the linear model to be related to the response
variable via a link function and by allowing the magnitude of the variance of each measurement to
be a function of its predicted value.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|epochs|The number of complete pass through the training data.|Integer|N|Default: 10.Range: [1, $\infty$).|
|learning_rate|The step size at each iteration in one iteration.|Float|N|Default: 0.1.Range: (0.0, $\infty$).|
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|link_type|link function type|String|Y|Allowed: ['Logit', 'Log', 'Reciprocal', 'Identity'].|
|label_dist_type|label distribution type|String|Y|Allowed: ['Bernoulli', 'Poisson', 'Gamma', 'Tweedie'].|
|tweedie_power|Tweedie distribution power parameter|Float|N|Default: 1.0.Range: [0.0, 2.0].|
|dist_scale|A guess value for distribution's scale|Float|N|Default: 1.0.Range: [1.0, $\infty$).|
|iter_start_irls|run a few rounds of IRLS training as the initialization of w, 0 disable|Integer|N|Default: 0.Range: [0, $\infty$).|
|decay_epoch|decay learning interval|Integer|N|Default: 0.Range: [0, $\infty$).|
|decay_rate|decay learning rate|Float|N|Default: 0.0.Range: [0.0, 1.0).|
|optimizer|which optimizer to use: IRLS(Iteratively Reweighted Least Squares) or SGD(Stochastic Gradient Descent)|String|Y|Allowed: ['SGD', 'IRLS'].|
|l2_lambda|L2 regularization term|Float|N|Default: 0.1.Range: [0.0, $\infty$).|
|infeed_batch_size_limit|size of a single block, default to 8w * 100. increase the size will increase memory cost, but may decrease running time. Suggested to be as large as possible. (too large leads to OOM)|Integer|N|Default: 8000000.Range: [1000, 8000000].|
|fraction_of_validation_set|fraction of training set to be used as the validation set. ineffective for 'weight' stopping_metric|Float|N|Default: 0.2.Range: (0.0, 1.0).|
|random_state|random state for validation split|Integer|N|Default: 1212.Range: [0, $\infty$).|
|stopping_metric|use what metric as the condition for early stop? Must be one of ['deviance', 'MSE', 'RMSE', 'AUC', 'weight']. only logit link supports AUC metric (note that AUC is very, very expansive in MPC)|String|N|Default: deviance.Allowed: ['deviance', 'MSE', 'RMSE', 'AUC', 'weight'].|
|stopping_rounds|If the model is not improving for stopping_rounds, the training process will be stopped, for 'weight' stopping metric, stopping_rounds is fixed to be 1|Integer|N|Default: 0.Range: [0, 100].|
|stopping_tolerance|the model is considered as not improving, if the metric is not improved by tolerance over best metric in history. If metric is 'weight' and tolerance == 0, then early stop is disabled.|Float|N|Default: 0.001.Range: [0.0, 1.0).|
|report_metric|Whether to report the value of stopping metric. Only effective if early stop is enabled. If this option is set to true, metric will be revealed and logged.|Boolean|N|Default: False.|
|use_high_precision_exp|If you do not know the details of this parameter, please do not modify this parameter! If this option is true, glm training and prediction will use a high-precision exp approx, but there will be a large performance drop. Otherwise, use high performance exp approx, There will be no significant difference in model performance. However, prediction bias may occur if the model is exported to an external system for use.|Boolean|N|Default: False.|
|exp_iters|If you do not know the details of this parameter, please do not modify this parameter! Specify the number of iterations of exp taylor approx, Only takes effect when use_high_precision_exp is false. Increasing this value will improve the accuracy of exp approx, but will quickly degrade performance.|Integer|N|Default: 8.Range: [4, 32].|
|report_weights|If this option is set to true, model will be revealed and model details are visible to all parties|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/train_dataset/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. |
|input/train_dataset/offset|Specify a column to use as the offset|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Max column number to select(inclusive): 1. |
|input/train_dataset/weight|Specify a column to use for the observation weights|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Max column number to select(inclusive): 1. |
|input/train_dataset/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.ss_glm']||
|report|If report_weights is true, report model details|['sf.report']||

### ss_sgd_train


Component version: 0.0.1

Train both linear and logistic regression
linear models for vertical partitioning dataset with mini batch SGD training solver by using secret sharing.
- SS-SGD is short for secret sharing SGD training.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|epochs|The number of complete pass through the training data.|Integer|N|Default: 10.Range: [1, $\infty$).|
|learning_rate|The step size at each iteration in one iteration.|Float|N|Default: 0.1.Range: (0.0, $\infty$).|
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|sig_type|Sigmoid approximation type.|String|N|Default: t1.Allowed: ['real', 't1', 't3', 't5', 'df', 'sr', 'mix'].|
|reg_type|Regression type|String|N|Default: logistic.Allowed: ['linear', 'logistic'].|
|penalty|The penalty(aka regularization term) to be used.|String|N|Default: None.Allowed: ['None', 'l1', 'l2'].|
|l2_norm|L2 regularization term.|Float|N|Default: 0.5.Range: [0.0, $\infty$).|
|eps|If the change rate of weights is less than this threshold, the model is considered to be converged, and the training stops early. 0 to disable.|Float|N|Default: 0.001.Range: [0.0, $\infty$).|
|report_weights|If this option is set to true, model will be revealed and model details are visible to all parties|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/train_dataset/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. |
|input/train_dataset/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.ss_sgd']||
|report|If report_weights is true, report model details|['sf.report']||

### ss_xgb_train


Component version: 0.0.1

This method provides both classification and regression tree boosting (also known as GBDT, GBM)
for vertical partitioning dataset setting by using secret sharing.
- SS-XGB is short for secret sharing XGB.
- More details: https://arxiv.org/pdf/2005.08479.pdf
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|num_boost_round|Number of boosting iterations.|Integer|N|Default: 10.Range: [1, $\infty$).|
|max_depth|Maximum depth of a tree.|Integer|N|Default: 5.Range: [1, 16].|
|learning_rate|Step size shrinkage used in updates to prevent overfitting.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|objective|Specify the learning objective.|String|N|Default: logistic.Allowed: ['linear', 'logistic'].|
|reg_lambda|L2 regularization term on weights.|Float|N|Default: 0.1.Range: [0.0, 10000.0].|
|subsample|Subsample ratio of the training instances.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|colsample_by_tree|Subsample ratio of columns when constructing each tree.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|sketch_eps|This roughly translates into O(1 / sketch_eps) number of bins.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|base_score|The initial prediction score of all instances, global bias.|Float|N|Default: 0.0.Range: [-10.0, 10.0].|
|seed|Pseudorandom number generator seed.|Integer|N|Default: 42.Range: [0, $\infty$).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/train_dataset/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. |
|input/train_dataset/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table train_dataset. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.ss_xgb']||

## model

### model_export


Component version: 0.0.1

The model_export component supports converting and packaging the rule files generated by preprocessing and postprocessing components, as well as the model files generated by model operators, into a Secretflow-Serving model package. The list of components to be exported must contain exactly one model train or model predict component, and may include zero or multiple preprocessing and postprocessing components.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|model_name|model's name|String|Y||
|model_desc|Describe what the model does|String|N|Default: .|
|input_datasets|The input data IDs for all components to be exported. Their order must remain consistent with the sequence in which the components were executed.|String List|Y||
|output_datasets|The output data IDs for all components to be exported. Their order must remain consistent with the sequence in which the components were executed.|String List|Y||
|component_eval_params|The eval parameters (in JSON format) for all components to be exported. Their order must remain consistent with the sequence in which the components were executed.|String List|Y||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|package_output|output tar package uri|['sf.serving.model']||
|report|report dumped model's input schemas|['sf.report']||

## postprocessing

### score_card_transformer


Component version: 0.0.1

Transform the predicted result (a probability value) produced by the logistic regression model into a more understandable score (for example, a score of up to 1000 points)
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|positive|Value for positive cases.|Integer|Y|Allowed: [0, 1].|
|predict_score_name||String|Y||
|scaled_value|Set a benchmark score that can be adjusted for specific business scenarios|Integer|Y||
|odd_base|the odds value at given score baseline, odds = p / (1-p)|Float|Y||
|pdo|points to double the odds|Float|Y||
|min_score|An integer of [0,999] is supported|Integer|N|Default: 0.Range: [0, 999].|
|max_score|An integer of [1,1000] is supported|Integer|N|Default: 1000.Range: [1, 1000].|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|predict result table|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/predict_name||String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|output table|['sf.table.individual']||

## preprocessing

### binary_op


Component version: 0.0.2

Perform binary operation binary_op(f1, f2) and assign the result to f3, f3 can be new or old. Currently f1, f2 and f3 all belong to a single party.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|binary_op|What kind of binary operation we want to do, currently only supports +, -, *, /|String|N|Default: +.Allowed: ['+', '-', '*', '/'].|
|new_feature_name|Name of the newly generated feature.|String|Y||
|as_label|If True, the generated feature will be marked as label in schema.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|in_ds|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/in_ds/f1|Feature 1 to operate on.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |
|input/in_ds/f2|Feature 2 to operate on.|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|out_ds|Output vertical table.|['sf.table.vertical_table']||
|out_rules|feature gen rule|['sf.rule.preprocessing']||

### case_when


Component version: 0.0.1

case_when
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|rules|input CaseWhen rules|Special type. SecretFlow customized Protocol Buffers message.|Y||

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_dataset|Input vertical table.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_dataset|output_dataset|['sf.table.vertical_table']||
|out_rules|case when substitution rule|['sf.rule.preprocessing']||

### cast


Component version: 0.0.1

For conversion between basic data types, such as converting float to string.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|astype|single-choice, options available are string, integer, float|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|The input table|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/input_ds/columns|Multiple-choice, options available are string, integer, float, boolean|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|The output table|['sf.table.vertical_table']||
|output_rules|The output rules|['sf.rule.preprocessing']||

### feature_calculate


Component version: 0.0.1

Generate a new feature by performing calculations on an origin feature
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|rules|input CalculateOpRules rules|Special type. SecretFlow customized Protocol Buffers message.|Y||

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|in_ds|Input vertical table|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/in_ds/features|Feature(s) to operate on|String List(Set value with other Component Attributes)|You need to select some columns of table in_ds. Min column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|out_ds|output_dataset|['sf.table.vertical_table']||
|out_rules|feature calculate rule|['sf.rule.preprocessing']||

### fillna


Component version: 0.0.1

fillna
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|strategy|The imputation strategy. If "mean", then replace missing values using the mean along each column. Can only be used with numeric data. If "median", then replace missing values using the median along each column. Can only be used with numeric data. If "most_frequent", then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned. If "constant", then replace missing values with fill_value. Can be used with strings or numeric data.|String|N|Default: constant.Allowed: ['mean', 'median', 'most_frequent', 'constant'].|
|missing_value_type|type of missing value. general_na type indicates that only np.nan, None or pandas.NA will be treated as missing values. When the type is not general_na, the type casted missing_value_type(missing_value) will also be treated as missing value as well, in addition to general_na values.|String|N|Default: general_na.Allowed: ['general_na', 'str', 'int', 'float'].|
|missing_value|Which value should be treat as missing_value? If missing value type is 'general_na', this field will be ignored, and any np.nan, pd.NA, etc value will be treated as missing value. Otherwise, the type casted missing_value_type(missing_value) will also be treated as missing value as well, in addition to general_na values. In case the cast is not successful, general_na will be used instead. default value is 'custom_missing_value'.|String|N|Default: custom_missing_value.|
|fill_value_int|For int type data. If method is 'constant' use this value for filling null.|Integer|N|Default: 0.|
|fill_value_float|For float type data. If method is 'constant' use this value for filling null.|Float|N|Default: 0.0.|
|fill_value_str|For str type data. If method is 'constant' use this value for filling null.|String|N|Default: .|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/input_dataset/fill_na_features|Features to fill.|String List(Set value with other Component Attributes)|You need to select some columns of table input_dataset. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|out_ds|Output vertical table.|['sf.table.vertical_table']||
|out_rules|fill value rule|['sf.rule.preprocessing']||

### onehot_encode


Component version: 0.0.3

onehot_encode
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|drop|drop unwanted category based on selection|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|min_frequency|Specifies the minimum frequency below which a category will be considered infrequent, [0, 1), 0 disable|Float|N|Default: 0.0.Range: [0.0, 1.0).|
|report_rules|Whether to report rule details|Boolean|N|Default: True.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_dataset|Input vertical table.|['sf.table.vertical_table']|Pleae fill in extra table attributes.|
|input/input_dataset/features|Features to encode.|String List(Set value with other Component Attributes)|You need to select some columns of table input_dataset. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_dataset|output_dataset|['sf.table.vertical_table']||
|out_rules|onehot rule|['sf.rule.preprocessing']||
|report|report rules details if report_rules is true|['sf.report']||

### substitution


Component version: 0.0.2

unified substitution component
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_dataset|Input vertical table.|['sf.table.vertical_table']||
|input_rules|Input preprocessing rules|['sf.rule.preprocessing']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_dataset|output_dataset|['sf.table.vertical_table']||

### vert_bin_substitution


Component version: 0.0.1

Substitute datasets' value by bin substitution rules.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Vertical partitioning dataset to be substituted.|['sf.table.vertical_table']||
|bin_rule|Input bin substitution rule.|['sf.rule.binning']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_data|Output vertical table.|['sf.table.vertical_table']||

## stats

### groupby_statistics


Component version: 0.0.3

Get a groupby of statistics, like pandas groupby statistics.
Currently only support VDataframe.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|aggregation_config|input groupby aggregation config|Special type. SecretFlow customized Protocol Buffers message.|Y||
|max_group_size|The maximum number of groups allowed|Integer|N|Default: 10000.Range: (0, 10001).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input table.|['sf.table.vertical_table', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_data/by|by what columns should we group the values|String List(Set value with other Component Attributes)|You need to select some columns of table input_data. Min column number to select(inclusive): 1. Max column number to select(inclusive): 4. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output groupby statistics report.|['sf.report']||

### ss_pearsonr


Component version: 0.0.1

Calculate Pearson's product-moment correlation coefficient for vertical partitioning dataset
by using secret sharing.
- For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_data/feature_selects|Specify which features to calculate correlation coefficient with. If empty, all features will be used|String List(Set value with other Component Attributes)|You need to select some columns of table input_data. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output Pearson's product-moment correlation coefficient report.|['sf.report']||

### ss_vif


Component version: 0.0.1

Calculate Variance Inflation Factor(VIF) for vertical partitioning dataset
by using secret sharing.
- For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_data/feature_selects|Specify which features to calculate VIF with. If empty, all features will be used.|String List(Set value with other Component Attributes)|You need to select some columns of table input_data. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output Variance Inflation Factor(VIF) report.|['sf.report']||

### stats_psi


Component version: 0.0.1

population stability index.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_base_data|Input base vertical table.|['sf.table.vertical_table', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_base_data/feature_selects|which features should be binned.|String List(Set value with other Component Attributes)|You need to select some columns of table input_base_data. Min column number to select(inclusive): 1. |
|input_test_data|Input test vertical table.|['sf.table.vertical_table', 'sf.table.individual']||
|bin_rule|Input bin rule.|['sf.rule.binning']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output population stability index.|['sf.report']||

### table_statistics


Component version: 0.0.2

Get a table of statistics,
including each column's
1. datatype
2. total_count
3. count
4. count_na
5. na_ratio
6. min
7. max
8. mean
9. var
10. std
11. sem
12. skewness
13. kurtosis
14. q1
15. q2
16. q3
17. moment_2
18. moment_3
19. moment_4
20. central_moment_2
21. central_moment_3
22. central_moment_4
23. sum
24. sum_2
25. sum_3
26. sum_4
- moment_2 means E[X^2].
- central_moment_2 means E[(X - mean(X))^2].
- sum_2 means sum(X^2).
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input table.|['sf.table.vertical_table', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_data/features|perform statistics on these columns|String List(Set value with other Component Attributes)|You need to select some columns of table input_data. Min column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output table statistics report.|['sf.report']||
