import einml
from einml.prelude import *
import official.nlp.modeling.models.t5 as t5

### Simple character-based language model, demonstrating the use of seqio for mixed pre-training tasks and using the default T5 transformer implementation from tf models

# Load text-only english wikipedia dataset
ds = datasets.load.load_dataset("wikipedia", "20220301.en")



# Create a model
