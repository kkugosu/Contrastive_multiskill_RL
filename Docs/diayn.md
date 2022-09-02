
## DIAYN

we can learn lots of skills which can be base of unsupervised meta learning

this paper use fixed prior term to avoid model collapse. which can reduce number of skills very much. ex) vic

this paper use sac to make policy to search extensively and sac can make policy to avoid distributional shift problem.

but i don't know if sac is necessary because we already maximize IG from state.

below is rough process of diayn

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ simulate \ in \ environment \ while \ memory \ become \ full$$

$$3. \ compute \ q_\phi(z | s_t) \ from \ discriminator \ q_\phi$$

$$4. \ improve \ value \ log(q_\phi(z | s_t)) - log(p(z)) \ by \ update \ \phi$$

$$5. \ set \ reward \ as \ log(q_\phi(z | s_t)) - log(p(z))$$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$
