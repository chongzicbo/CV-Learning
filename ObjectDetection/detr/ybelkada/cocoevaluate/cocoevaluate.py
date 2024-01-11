# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""TODO: Add a description here."""

import evaluate
import datasets
import pyarrow as pa

from .coco_utils import CocoEvaluator, get_coco_api_from_dataset

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:module,
title = {A great new module},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the module here
_DESCRIPTION = """\
This new module is designed to solve this great ML task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_module = evaluate.load("my_new_module")
    >>> results = my_new_module.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"

# lists - summarize long lists similarly to NumPy
# arrays/tensors - let the frameworks control formatting
def summarize_if_long_list(obj):
    if not type(obj) == list or len(obj) <= 6:
        return f"{obj}"

    def format_chunk(chunk):
        return ", ".join(repr(x) for x in chunk)

    return f"[{format_chunk(obj[:3])}, ..., {format_chunk(obj[-3:])}]"

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class COCOEvaluate(evaluate.Metric):
    """TODO: Short description of my evaluation module."""
    def __init__(self, coco, iou_types=['bbox'], **kwargs):
        super().__init__(**kwargs)
        self.coco_evaluator = CocoEvaluator(coco, iou_types)


    def _info(self):
        # TODO: Specifies the evaluate.EvaluationModuleInfo object
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': [
                        datasets.Features(
                            {
                                'scores': datasets.Sequence(datasets.Value("float")),
                                'labels': datasets.Sequence(datasets.Value("int64")),
                                'boxes': datasets.Sequence(datasets.Sequence(datasets.Value("float"))),
                            })
                    ]
                    ,
                'references': [
                        datasets.Features(
                        {
                            'size': datasets.Sequence(datasets.Value("float")),
                            'image_id': datasets.Sequence(datasets.Value("int64")),
                            'boxes': datasets.Sequence(datasets.Sequence(datasets.Value("float"))),
                            'class_labels': datasets.Sequence(datasets.Value("int64")),
                            'iscrowd': datasets.Sequence(datasets.Value("int64")),
                            'orig_size': datasets.Sequence(datasets.Value("float")),
                            'area': datasets.Sequence(datasets.Value("float")),
                            
                        }
                    )
                ],

            }),
            # Homepage of the module for documentation
            homepage="http://module.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_module"],
            reference_urls=["http://path.to.reference.url/new_module"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        # TODO: Download external resources if needed
        pass

    def _preprocess(self, predictions):
        """Optional: preprocess the predictions and references before computing the scores"""
        processed_predictions = []
        for pred in predictions:
            processed_pred = {}
            for key in pred.keys():
                processed_pred[key] = pred[key].detach().cpu().tolist()
            processed_predictions.append(processed_pred)
        return processed_predictions
    
    def add(self, *, prediction=None, reference=None, **kwargs):
        """Preprocesses the predictions and references and calls the function of the parent class."""
        if prediction is not None:
            prediction = self._preprocess(prediction)
        if reference is not None:
            reference = self._preprocess(reference)
        super().add(prediction=prediction, references=reference, **kwargs)

    def _compute(self, predictions, references):
        """Returns the scores"""
        for pred, ref in zip(predictions, references):
            res = {}
            for target, output in zip(ref, pred):
                res[target["image_id"][0]] = output
            self.coco_evaluator.update(res)
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()

        self.coco_evaluator.summarize()

        stats = self.coco_evaluator.get_results()

        return stats