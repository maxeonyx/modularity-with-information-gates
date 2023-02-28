"""
# LiSTran
## Linear, Scaled Transformer

LiSTran is a transformer that uses linear attention and careful use of scaling to maintain
quality while reducing compute requirements. Instead of using softmax to compute attention,
it uses a linear function. It uses a dot product for attention, but instead of a
softmax-normalized convex sum it uses a linear combination with constant coefficients according
to the number of inputs, which varies depending on the mask.
"""

from einml.prelude import *
from official.nlp.modeling.models import t5
