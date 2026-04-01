import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torch.utils.data.datapipes.iter import IterableWrapper     # api changed
from itertools import chain
from typing import Optional,List

import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
import swanlab

target_modules = ["q_proj", "q_proj"]
