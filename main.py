import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

from tensorflow import keras
from tensorflow.keras import Model, Input, layers

import tensorflow_hub as hub
import tensorflow_models as tfm
import orbit
import seqio
import datasets # HuggingFace Datasets

### Simple character-based language model, demonstrating the use of seqio for mixed pre-training tasks and using the default T5 transformer implementation from tf models

# Load text-only english wikipedia dataset
from datasets import load_dataset
load_dataset("wikipedia", "20220301.en")

# Create a model

tfm