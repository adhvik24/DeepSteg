# The code in the src folder is taken from the github https://github.com/daniellerch/hstego. The libraries and all 
from Juniward import J_UNIWARD

# Embedding
jp = J_UNIWARD()
def embed(cover, secret, stego):
    jp.embed(cover, secret, stego)
